#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
  sharedUpdate(y);
}
/*{{{*/
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd z_pred = cartesianToPolar(x_);
	VectorXd y = z - z_pred;
  y(1) = normalizeAngle(y);
  sharedUpdate(y);
}

void KalmanFilter::UpdateStateTransition(const float dt) {
  F_(0,2) = dt;
  F_(1,3) = dt;
}

void KalmanFilter::UpdateProcessCovariance(const float dt,
    const int noise_ax,
    const int noise_ay) {

  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt3 * dt;
  Q_ << dt4/4 * noise_ax, 0, dt3/2 * noise_ax, 0,
     0, dt4/4 * noise_ay, 0, dt3/2 * noise_ay,
     dt3/2 * noise_ax, 0, dt2 * noise_ax, 0,
     0, dt3/2 * noise_ay, 0, dt2 * noise_ay;
}

float KalmanFilter::normalizeAngle(const VectorXd &y) {
  float px = cos(y(1));
  float py = sin(y(1));
  float n_theta = atan2(py,px);
  return n_theta;
}

MatrixXd KalmanFilter::cartesianToPolar(const VectorXd &x) {
	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);

	float rho = sqrt(px*px+py*py);
	float theta = atan2(py,px);
	float ro_dot = (px * vx + py * vy) / rho;

	VectorXd z_pred = VectorXd(3);
	z_pred << rho, theta, ro_dot;

  return z_pred;
}

void KalmanFilter::sharedUpdate(const VectorXd &y) {
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
