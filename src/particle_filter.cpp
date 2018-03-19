#include <random>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;

/** Evaluate a Gaussian distribution with the specified mean and standard deviation at the specified values. */
double gaussian(double mu, double std, double x)
{
    return exp(-pow((mu - x) / std, 2) / 2.0) / (sqrt(2.0 * M_PI) * std);
}

/** Evaluate the likelihood that an observation corresponds to a specific landmark.
 * Note that since you are passing in a list of standard deviations, we can assume that
 * the measurements are independent. */
double likelihood(const LandmarkObs &observation, const LandmarkObs &landmark, double std[])
{
    return gaussian(landmark.x, std[0], observation.x) * gaussian(landmark.y, std[1], observation.y);
}

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    /* Specify the number of particles we will be using and resize everything */
    num_particles = 10;
    particles.resize(num_particles);
    weights.resize(num_particles);

    /* Create the random distributions from which we will sample particles. */
    normal_distribution<double> x_distribution(x, std[0]);
    normal_distribution<double> y_distribution(y, std[1]);
    normal_distribution<double> theta_distribution(theta, std[2]);

    /* Finally, initialize all of our particles at random positions and headings.
     * Furthermore, set the weights according to their likelihood. */
    for (int i = 0; i < num_particles; ++i)
    {
        Particle particle;
        particle.id = i;
        particle.x = x_distribution(rng);
        particle.y = y_distribution(rng);
        particle.theta = theta_distribution(rng);
        particle.weight = gaussian(x, std[0], particle.x) * gaussian(y, std[1], particle.y) * gaussian(theta, std[2], particle.theta);
        particles[i] = particle;
        weights[i] = particle.weight;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std[], double velocity, double yaw_rate)
{
    normal_distribution<double> x_distribution(0, std[0]);
    normal_distribution<double> y_distribution(0, std[1]);
    normal_distribution<double> theta_distribution(0, std[2]);
    for (auto &particle : particles)
    {
        if (abs(yaw_rate) > 1e-5)
        {
            double scale = velocity / yaw_rate;
            double a = particle.theta + dt * yaw_rate;
            particle.x += scale * (sin(a) - sin(particle.theta));
            particle.y += scale * (-cos(a) + cos(particle.theta));
            particle.theta += dt * yaw_rate;
        }
        else
        {
            particle.x += velocity * dt * cos(particle.theta);
            particle.y += velocity * dt * sin(particle.theta);
            particle.theta += dt * yaw_rate;
        }
        /* Add random Gaussian noise */
        particle.x += x_distribution(rng);
        particle.y += y_distribution(rng);
        particle.theta += theta_distribution(rng);
    }
}

/** Transform the observation into map coordinates assuming that the specified particle made the observation. */
LandmarkObs transformToMapCoordinates(const LandmarkObs &observation, const Particle &particle)
{
    LandmarkObs transformed_observation;
    transformed_observation.x = particle.x + cos(particle.theta) * observation.x - sin(particle.theta) * observation.y;
    transformed_observation.y = particle.y + sin(particle.theta) * observation.x + cos(particle.theta) * observation.y;
    return transformed_observation;
}

/** Find the landmark closest to the specified observation.
 * NOTE: the observation and landmarks should be in the same coordinate system. */
LandmarkObs nearestLandmark(const LandmarkObs &observation, const Map &map_landmarks)
{

    double distance = -1;
    LandmarkObs nearest_landmark;
    for (auto &landmark : map_landmarks.landmark_list)
    {
        double dst = pow(observation.x - landmark.x_f, 2) + pow(observation.y - landmark.y_f, 2);
        if (dst < distance || distance < 0)
        {
            distance = dst;
            nearest_landmark.x = landmark.x_f;
            nearest_landmark.y = landmark.y_f;
        }
    }
    return nearest_landmark;
}

/**
 * The observations occur in the car coordinate system.
 * We want to find which landmark each corresponds to.
 * However, we don't know the relationship between the car's coordinate system and the map's coordinate system.
 * That is what we are trying to estimate.
 * You can think of the particle filter as maintaining a set of "possible" car locations.
 * So for each particle we must follow this process:
 *
 * Assuming that the car's position and orientation are at the particle's position and orientation,
 * transform the coordinates of the observations to the map coordinate system. Then find the nearest
 * neighbor matches between these transformed observations and the map. Once we have found the nearest
 * neighbors, we can evaluate how likely the set of measurements by using what we know about the
 * sensor statistics. The more likely the set of measurements is, the more likely that the particle
 * has a position and orientation similar to that of the actual car. So we update the particle's weight
 * to reflect this likelihood.
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
    for (int i = 0; i < num_particles; ++i)
    {

        /* Reset the weight for this particle */
        weights[i] = 1;
        for (auto &observation : observations)
        {

            /* Transform the observation to the map coordinate system. */
            LandmarkObs transformed_observation = transformToMapCoordinates(observation, particles[i]);

            /* Find the nearest landmark. */
            LandmarkObs nearest_landmark = nearestLandmark(transformed_observation, map_landmarks);

            /* Calculate the likelihood of the observation, and update the weight */
            weights[i] *= likelihood(transformed_observation, nearest_landmark, std_landmark);
        }
        particles[i].weight = weights[i];
    }
}

void ParticleFilter::resample()
{

    /* Calculate the sum of all of the weights */
    std::vector<double> cumulative_weights;
    cumulative_weights.reserve(num_particles);
    cumulative_weights.push_back(weights[0]);
    for (int i = 1; i < num_particles; ++i)
        cumulative_weights.push_back(cumulative_weights[i - 1] + weights[i]);

    /* Make space for the resampled particles */
    std::vector<Particle> resampled_particles;
    resampled_particles.resize(num_particles);

    /* Now generate the resampled particles by estimating a random number,
     * and then choosing the corresponding sample. */
    std::uniform_real_distribution<double> distribution(0, cumulative_weights.back());
    for (int i = 0; i < num_particles; ++i)
    {

        /* Generate a random number in the range [ 0, cumulative_weight ] */
        double rand = distribution(rng);

        /* Now find which bin the random number falls in. */
        int j = 0;
        while (cumulative_weights[j] <= rand)
        {
            ++j;
        }
        resampled_particles[i] = particles[j];
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
