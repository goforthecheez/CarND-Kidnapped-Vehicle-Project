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
#include <float.h>
#include <cmath>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    std::random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 50;
    for (int i = 0; i < num_particles; ++i) {
        weights.push_back(1.0);
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    // To make associating weights to particles easier later, re-ID the particles.
    std::vector<Particle> new_particles;
    // While we're at it, also invalidate the weights.
    std::fill(weights.begin(), weights.end(), 0.0);

    // Make a motion prediction for each particle.
    for (uint i = 0; i < particles.size(); ++i) {
        Particle new_particle;
        // Reset the particle's ID.
        new_particle.id = i;

        // Add noise after applying the noiseless control input.
        Particle& p = particles[i];
        if (fabs(yaw_rate) < FLT_EPSILON) {
            new_particle.x = p.x + velocity * cos(p.theta) * delta_t + dist_x(gen);
            new_particle.y = p.y + velocity * sin(p.theta) * delta_t + dist_y(gen);
            new_particle.theta = p.theta + dist_theta(gen);
        }
        else {
            new_particle.x = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
            new_particle.y = p.y + velocity/yaw_rate * (-cos(p.theta + yaw_rate * delta_t) + cos(p.theta)) + dist_y(gen);
            new_particle.theta = p.theta + yaw_rate * delta_t + dist_theta(gen);
        }

        // Finished with the motion prediction.
        new_particles.push_back(new_particle);
    }

    // Save the motion-updated particles.
    particles = new_particles;
}

double ParticleFilter::euclideanDistance(double x1, double x2, double y1, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs& obs : observations) {
        int closest_id = -1;
        double min_dist = DBL_MAX;
        for (LandmarkObs& pred : predicted) {
            double dist = euclideanDistance(obs.x, pred.x, obs.y, pred.y);
            if (dist < min_dist) {
                closest_id = pred.id;
                min_dist = dist;
            }
        }
        obs.id = closest_id;
    }
}

double ParticleFilter::multivariateGaussian(const LandmarkObs& obs, const LandmarkObs& pred,
                                            double std_landmark[]) {
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    double gauss_norm = 1/(2 * M_PI * sig_x * sig_y);
    double exponent = pow(obs.x - pred.x, 2)/(2 * pow(sig_x, 2)) + pow(obs.y - pred.y, 2)/(2 * pow(sig_y, 2));
    return gauss_norm * exp(-exponent);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Compute new weights for each particle.
	for (uint i = 0; i < particles.size(); ++i) {
	    Particle& p = particles[i];

	    // Create predicted landmarks list.
    	std::vector<LandmarkObs> predicted;
	    for (auto landmark : map_landmarks.landmark_list) {
	        if (euclideanDistance(p.x, landmark.x_f, p.y, landmark.y_f) < sensor_range) {
	            predicted.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
	        }
	    }
	    // If all landmarks are outside sensor range, this particle should receive 0 weight.
	    if (predicted.size() == 0) {
	        p.weight = 0.0;
	        weights[i] = 0.0;
	        continue;
	    }

	    // Convert observations to the map's coordinate system.
	    std::vector<LandmarkObs> converted_observations;
	    for (const LandmarkObs& obs : observations) {
    	    double x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
            double y_map = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
            converted_observations.push_back({-1, x_map, y_map});
        }

	    // Perform data association.
	    dataAssociation(predicted, converted_observations);

        // Compute likelihood and update weights.
        // Also store the landmark associations, so the simulator can display them.
        std::vector<int> associations;
	    std::vector<double> sense_x;
	    std::vector<double> sense_y;
        float likelihood = 1.0;
        for (LandmarkObs& obs : converted_observations) {
            associations.push_back(obs.id);
            sense_x.push_back(obs.x);
	        sense_y.push_back(obs.y);
            for (LandmarkObs& pred : predicted) {
                if (obs.id == pred.id) {
                    likelihood *= multivariateGaussian(obs, pred, std_landmark);
                    break;
                }
            }
        }
        SetAssociations(p, associations, sense_x, sense_y);
        p.weight = likelihood;
        weights[i] = likelihood;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
	default_random_engine gen(rd());
    discrete_distribution<int> dist(weights.begin(), weights.end());

    std::vector<Particle> new_particles;
    for (uint i = 0; i < particles.size(); ++i) {
        int picked_id = dist(gen);
        new_particles.push_back(particles[picked_id]);
    }
    particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
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
