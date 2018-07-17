/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits.h>
#include "particle_filter.h"

#define MAX_PARTICLES     50
#define DEFAULT_WEIGHT    1.0

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;

  num_particles = MAX_PARTICLES;

  // Create normal (Gaussian) distributions for x, y and theta
  normal_distribution<double> px(x, std[0]);
  normal_distribution<double> py(y, std[1]);
  normal_distribution<double> ptheta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p = {i, px(gen), py(gen), ptheta(gen), DEFAULT_WEIGHT};
    particles.push_back(p);
    weights.push_back(DEFAULT_WEIGHT);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  // Create normal (Gaussian) distributions for x, y and theta
  normal_distribution<double> noiseX(0, std_pos[0]);
  normal_distribution<double> noiseY(0, std_pos[1]);
  normal_distribution<double> noiseTheta(0, std_pos[2]);

  const double yaw = yaw_rate * delta_t;

  for (int i = 0; i < num_particles; i++) {
    const double theta = particles[i].theta;

    if (fabs(yaw_rate) < 0.001) {
      const double c = velocity * delta_t;
      particles[i].x += c * cos(theta);
      particles[i].y += c * sin(theta);
    }
    else {
      const double c = velocity / yaw_rate;
      particles[i].x += c * (sin(theta + yaw) - sin(theta));
      particles[i].y += c * (cos(theta) - cos(theta + yaw));
      particles[i].theta += yaw;
    }
    // Add noise to the particles
    particles[i].x += noiseX(gen);
    particles[i].y += noiseY(gen);
    particles[i].theta += noiseTheta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  double  minDistance, distance, diffX, diffY;
  int   index;

  for (int i = 0; i < observations.size(); i++) {
    minDistance = numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++) {
      diffX = predicted[j].x - observations[i].x;
      diffY = predicted[j].y - observations[i].y;
      distance = diffX * diffX + diffY * diffY;
      if (distance < minDistance) {
        minDistance = distance;
        index = j;
      }
    }
    observations[i].id = index;
  }
}

LandmarkObs ParticleFilter::vehicleToMapCoord(const LandmarkObs obs, const Particle p) {
  LandmarkObs mapCoord;

  const double obsX = obs.x;
  const double obsY = obs.y;
  const double theta = p.theta;

  mapCoord.x = obsX * cos(theta) - obsY * sin(theta) + p.x;
  mapCoord.y = obsX * sin(theta) + obsY * cos(theta) + p.y;
  mapCoord.id = obs.id;

  return mapCoord;
}

double ParticleFilter::mvGaussianProb(const double std_landmark[], const std::vector<LandmarkObs> mapObs, const std::vector<LandmarkObs> landmarks) {
  const double stdX = std_landmark[0];
  const double stdY = std_landmark[1];
  const double gaussNorm = 2.0 * M_PI * stdX * stdY;
  const double denomX = 2.0 * stdX * stdX;
  const double denomY = 2.0 * stdY * stdY;

  double mvGauss = 1.0;
  for (int j = 0; j < mapObs.size(); j++) {
    const int    id = mapObs[j].id;
    const double diffX = mapObs[j].x - landmarks[id].x;
    const double diffY = mapObs[j].y - landmarks[id].y;

    // Calculate multi-variate Gaussian distribution
    const double dx = (diffX * diffX) / denomX;
    const double dy = (diffY * diffY) / denomY;
    const double result = exp(-(dx + dy)) / gaussNorm;
    mvGauss *= result;
  }
  return mvGauss;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++) {
    const double px = particles[i].x;
    const double py = particles[i].y;
    const double ptheta = particles[i].theta;

    vector<LandmarkObs> mapObservations;
    vector<LandmarkObs> nearestLandmarks;

    // Transform each observation from vehicle coordinates to map coordinates
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs obs = vehicleToMapCoord(observations[j], particles[i]);
      mapObservations.push_back(obs);
    }

    // Find nearest landmark
    vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
    for (int k = 0; k < landmarks.size(); k++) {
      const double diffX = landmarks[k].x_f - px;
      const double diffY = landmarks[k].y_f - py;
      const double distance = (diffX * diffX) + (diffY * diffY);
      if (distance <= sensor_range * sensor_range) {
        LandmarkObs landmark = {landmarks[k].id_i, landmarks[k].x_f, landmarks[k].y_f};
        nearestLandmarks.push_back(landmark);
      }
    }
    dataAssociation(nearestLandmarks, mapObservations);

    const double mvGaussian = mvGaussianProb(std_landmark, mapObservations, nearestLandmarks);
    // Update particle weights with combined multi-variate Gaussian distribution
    particles[i].weight = mvGaussian;
    weights[i] = mvGaussian;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  // Vector for new particles
  vector<Particle> newParticles(num_particles);
  discrete_distribution<int> index(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++) {
    newParticles[i] = particles[index(gen)];
  }

  // Replace old particles with the resampled particles
  //particles = newParticles;
  particles = move(newParticles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
