// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include "helper/BasicEigenTypes.hpp"
#include "raisim/World.hpp"
#include "helper/controlHelper.hpp"
#include "helper/neuralNet.hpp"

namespace raisim {

class raibotController {

 public:

  enum class RewardType : int {
    CMDLINEAR = 1,
    CMDANGULAR,
    TORQUE,
    AIRTIME,
    JOINTSPEED,
    FOOTSLIP,
    FOOTCLEARANCE,
    ORIENTATION,
    SMOOTHNESS,
    SMOOTHNESS2,
    NOMINALPOS,
    JOINTACCEL,
    AVOIDANCE,
    FOOTFORCE,
    JOINTLIMIT
  };

  bool create(raisim::World *world) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    raibot->setControlMode(ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// get robot data
    gcDim_ = raibot->getGeneralizedCoordinateDim();
    gvDim_ = raibot->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.4725, 1, 0.0, 0.0, 0.0,
                0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672, 0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672;
    raibot->setState(gc_init_, gv_init_);

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    raibot->setPdGains(jointPgain, jointDgain);
    raibot->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 36; estDim_=8; hvDim_ = 23;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_); estTarget_.setZero(estDim_); hvDouble_.setZero(hvDim_);
    command_.setZero(); airTime_.setZero(); preTarget_.setZero(actionDim_); pre2Target_.setZero(actionDim_);
    preJointVel_.setZero(nJoints_); preTarget_.setZero(actionDim_); pre2Target_.setZero(actionDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// indices of links that should not make contact with ground
    footIndices_.push_back(raibot->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(raibot->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(raibot->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(raibot->getBodyIdx("RH_SHANK"));
    footFrameIndices_.push_back(raibot->getFrameIdxByName("LF_S2F"));
    footFrameIndices_.push_back(raibot->getFrameIdxByName("RF_S2F"));
    footFrameIndices_.push_back(raibot->getFrameIdxByName("LH_S2F"));
    footFrameIndices_.push_back(raibot->getFrameIdxByName("RH_S2F"));
    footPosCur_.resize(footFrameIndices_.size());
    footVelCur_.resize(footFrameIndices_.size());
    footNormalForce_.setZero();
    footSlippage_ = 0.; groundHeight_ = 0.;
    nearFootHeightMap_.setZero(16);

    nominalJointPosWeight_.setZero(nJoints_);
    nominalJointPosWeight_ << 1.414, 1., 1., 1.414, 1., 1., 1.414, 1., 1., 1.414, 1., 1.;

    updateObservation(world);

    positiveRwdTag_ = {"cmd_linear", "cmd_angular", "air_time_rew"};
    negativeRwdTag_ = {"joint_torque_rew", "joint_speed", "foot_slip", "foot_clearance", "orientation", "smoothness",
                       "smoothness2", "nominal_joint_pos", "joint_accel", "motion_avoidance", "foot_force", "joint_limit"};
    positiveRwd_.resize((int)positiveRwdTag_.size());
    negativeRwd_.resize((int)negativeRwdTag_.size());

    stepDataTag_.insert(stepDataTag_.end(), positiveRwdTag_.begin(), positiveRwdTag_.end());
    stepDataTag_.insert(stepDataTag_.end(), negativeRwdTag_.begin(), negativeRwdTag_.end());
    stepData_.resize((int)stepDataTag_.size());

    return true;
  }

  bool init(raisim::World *world) { return true; }

  bool reset(raisim::World *world) {
    command_ << 0., 0., 0.;
    updateObservation(world);
    return true;
  }

  bool reset_random(raisim::World *world, raisim::HeightMap *heightMap, const double &c_f) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    c_f_ = c_f;
    airTime_.setZero(); stanceTime_.setZero();

    gc_ = noisyGC(gc_init_, normDist_, gen_);
    gv_ = noisyGV(gv_init_, normDist_, gen_);
    if (standingMode_) gv_ *= 2.0;

    raibot->setGeneralizedCoordinate(gc_);
    double shift=-100, delta;
    for (int i = 0; i < footFrameIndices_.size(); ++i) {
      raibot->getFramePosition(footFrameIndices_[i], footPosCur_[i]);
      delta = heightMap->getHeight(footPosCur_[i](0), footPosCur_[i](1)) - footPosCur_[i](2);
      shift = shift > delta ? shift : delta;
    }
    gc_[2] += shift + 0.025 + 0.001;
    raibot->setState(gc_, gv_);

    for (int i = 0; i < footFrameIndices_.size(); ++i)
      raibot->getFramePosition(footFrameIndices_[i], footPosCur_[i]);
    groundHeight_ = heightMap->getHeight(gc_[0], gc_[1]);
    return true;
  }

  bool begin(raisim::World *world) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    standingMode_=false;
    /// command generation
    if (uniDist_(gen_) < 0.6666) {
      command_ << eggSampler(uniDist_(gen_), uniDist_(gen_), 2.5, 1.5, 1.5),
                  uniDist_(gen_) * 1.5; // yaw turning
    } else {
      command_ << ovalSampler(uniDist_(gen_), uniDist_(gen_), 2.5, 1.5, 1.5),
                  uniDist_(gen_) * 1.5; // yaw turning
    }
    if (command_.norm() < 0.3) { command_.setZero(); standingMode_=true; }

    /// external forces and torques
    raibot->clearExternalForcesAndTorques();
    raibot->setExternalForce(raibot->getBodyIdx("base"),
                             {30. * uniDist_(gen_), 30. * uniDist_(gen_), 15. * uniDist_(gen_)});
    raibot->setExternalTorque(raibot->getBodyIdx("base"),
                              {2. * uniDist_(gen_), 2. * uniDist_(gen_), 1. * uniDist_(gen_)});

    /// update observations
    updateObservation(world);
    updateEstTarget();
    updateHyperVision();
    return true;
  }

  void beforeStep() {
    contactState_.clear();
    footPos_.clear(); footVel_.clear();
    footNormalForce_.setZero();
    jointLimitMetric_ = 0;

    /// store previous variables
    preJointVel_ = gv_.tail(12);
    pre2Target_ = preTarget_;
    preTarget_ = pTarget12_;
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec>& action) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    raibot->setControlMode(ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    raibot->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  void accumulateSim(raisim::World *world, raisim::HeightMap *heightMap) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    footContactMask_.clear();
    contactStateCur_.clear();
    inContact_ = -1;

    /// find local body indices of contacting  foot
    for (auto& contact: raibot->getContacts()) {
      if (contact.skip()) continue;
      if (inContact_ == contact.getlocalBodyIndex()) {
        if(idx_ != footIndices_.size()) footNormalForce_[(int)idx_] += contact.getImpulse().e()[2];
        continue;
      }
      inContact_ = contact.getlocalBodyIndex();
      idx_ = std::find(footIndices_.begin(), footIndices_.end(), inContact_) - footIndices_.begin();
      if (idx_ == footIndices_.size()) continue;
      footContactMask_.push_back(inContact_);
      footNormalForce_[(int)idx_] += contact.getImpulse().e()[2];
    }

    /// air time calculation
    for (int i = 0; i < footIndices_.size(); ++i) {
      if (std::find(footContactMask_.begin(), footContactMask_.end(), footIndices_[i]) == footContactMask_.end()) {
        airTime_[i] += world->getTimeStep(); stanceTime_[i] = 0.; contactStateCur_.push_back(false);
      } else {
        stanceTime_[i] += world->getTimeStep(); airTime_[i] = 0.; contactStateCur_.push_back(true);
      }
    } contactState_.push_back(contactStateCur_);

    /// store foot pos and vel
    for (int i = 0; i < footFrameIndices_.size(); ++i) {
      raibot->getFramePosition(footFrameIndices_[i], footPosCur_[i]);
      raibot->getFrameVelocity(footFrameIndices_[i], footVelCur_[i]);
      footPosCur_[i](2) -= heightMap->getHeight(footPosCur_[i](0), footPosCur_[i](1));
      if (std::abs(footPosCur_[i](0)) > heightMap->getXSize()/2. || std::abs(footPosCur_[i](1)) > heightMap->getYSize()/2.)
        footPosCur_[i](2) += heightMap->getHeight(footPosCur_[i](0), footPosCur_[i](1));
    }
    footPos_.push_back(footPosCur_);
    footVel_.push_back(footVelCur_);

    /// joint limit metric
    gc_ = raibot->getGeneralizedCoordinate().e();
    for (int i = 9; i < gcDim_; i=i+3)
      if(gc_[i] < -2.4686 || gc_[i] > -0.0272)
        ++jointLimitMetric_;
  }

  void afterStep(raisim::World *world, raisim::HeightMap *heightMap) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// airtime specification
    airtimeTotal_ = 0.;
    for (int i = 0; i < 4; ++i) {
      if (standingMode_)
        airtimeTotal_ += std::min(std::max(stanceTime_[i] - airTime_[i], -0.3), 0.3);
      else {
        if (airTime_[i] < 0.30)
          airtimeTotal_ += std::min(airTime_[i], 0.25);
        else if (stanceTime_[i] < 0.30)
          airtimeTotal_ += std::min(stanceTime_[i], 0.25);
      }
    }

    /// foot-variables accumulator
    footSlippage_ = 0.; footClearance_ = 0.;
    for (int t = 0; t < contactState_.size(); ++t) {
      for (int i = 0; i < footFrameIndices_.size(); ++i) {
        if (contactState_[t][i])
          footSlippage_ += footVel_[t][i].e().head(2).squaredNorm();
        else
          footClearance_ += std::pow(footPos_[t][i].e()[2] - 0.20, 2) * footVel_[t][i].e().head(2).norm();
      }
    } footSlippage_ /= (int)contactState_.size(); footClearance_ /= (int)contactState_.size();

    /// footNormalForce Squared Sum
    footNormalForceSS_ = footNormalForce_.squaredNorm();

    /// ground height
    raibot->getState(gc_, gv_);
    groundHeight_ = heightMap->getHeight(gc_[0], gc_[1]);
    if (std::abs(gc_[0]) > heightMap->getXSize()/2. || std::abs(gc_[1]) > heightMap->getYSize()/2.)
      groundHeight_ = 0.;

    /// near foot height map
    for (int i = 0; i < nearFootHeightMap_.size(); i = i + 4) {
      nearFootHeightMap_[i + 0] = footPosCur_[i/4](2) - heightMap->getHeight((footPosCur_[i/4].e() + 0.10 * rot_.e().row(0).transpose())[0], footPosCur_[i/4](1));
      nearFootHeightMap_[i + 1] = footPosCur_[i/4](2) - heightMap->getHeight(footPosCur_[i/4](0), (footPosCur_[i/4].e() + 0.10 * rot_.e().row(1).transpose())[1]);
      nearFootHeightMap_[i + 2] = footPosCur_[i/4](2) - heightMap->getHeight((footPosCur_[i/4].e() - 0.10 * rot_.e().row(0).transpose())[0], footPosCur_[i/4](1));
      nearFootHeightMap_[i + 3] = footPosCur_[i/4](2) - heightMap->getHeight(footPosCur_[i/4](0), (footPosCur_[i/4].e() - 0.10 * rot_.e().row(1).transpose())[1]);
      if (std::abs((footPosCur_[i/4].e() + 0.10 * rot_.e().row(0).transpose())[0]) > heightMap->getXSize()/2. || std::abs(footPosCur_[i/4](1)) > heightMap->getYSize()/2.) 
        nearFootHeightMap_[i + 0] += heightMap->getHeight((footPosCur_[i/4].e() + 0.10 * rot_.e().row(0).transpose())[0], footPosCur_[i/4](1));
      if (std::abs(footPosCur_[i/4](0)) > heightMap->getXSize()/2. || std::abs((footPosCur_[i/4].e() + 0.10 * rot_.e().row(1).transpose())[1]) > heightMap->getYSize()/2.)
        nearFootHeightMap_[i + 1] += heightMap->getHeight(footPosCur_[i/4](0), (footPosCur_[i/4].e() + 0.10 * rot_.e().row(1).transpose())[1]);
      if (std::abs((footPosCur_[i/4].e() - 0.10 * rot_.e().row(0).transpose())[0]) > heightMap->getXSize()/2. || std::abs(footPosCur_[i/4](1)) > heightMap->getYSize()/2.)
        nearFootHeightMap_[i + 2] += heightMap->getHeight((footPosCur_[i/4].e() - 0.10 * rot_.e().row(0).transpose())[0], footPosCur_[i/4](1));
      if (std::abs(footPosCur_[i/4](0)) > heightMap->getXSize()/2. || std::abs((footPosCur_[i/4].e() - 0.10 * rot_.e().row(1).transpose())[1]) > heightMap->getYSize()/2.)
        nearFootHeightMap_[i + 3] += heightMap->getHeight(footPosCur_[i/4](0), (footPosCur_[i/4].e() - 0.10 * rot_.e().row(1).transpose())[1]);
    }
    
    updateObservation(world);
    updateEstTarget();
    updateHyperVision();
  }

  void updateReward(raisim::World *world, const std::map<RewardType, float>& rewardCoeff) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    double cmdLinear = rewardCoeff.at(RewardType::CMDLINEAR) * std::exp(-1. * (command_.head(2) - bodyLinearVel_.head(2)).squaredNorm());
    double cmdAngular = rewardCoeff.at(RewardType::CMDANGULAR) * std::exp(-1.5 * pow(command_(2) - bodyAngularVel_(2), 2));
    double jointTorqueReward = rewardCoeff.at(RewardType::TORQUE) * raibot->getGeneralizedForce().squaredNorm();
    double airTimeReward = rewardCoeff.at(RewardType::AIRTIME) * airtimeTotal_;
    double jointSpeedReward = rewardCoeff.at(RewardType::JOINTSPEED) * gv_.tail(12).squaredNorm();
    double footSlipReward = rewardCoeff.at(RewardType::FOOTSLIP) * footSlippage_;
    double footClearanceReward = rewardCoeff.at(RewardType::FOOTCLEARANCE) * footClearance_;
    double orientationReward = rewardCoeff.at(RewardType::ORIENTATION) * std::pow(std::acos(rot_(8)), 2);
    double smoothnessReward = rewardCoeff.at(RewardType::SMOOTHNESS) * (pTarget12_ - preTarget_).squaredNorm();
    double smoothness2Reward = rewardCoeff.at(RewardType::SMOOTHNESS2) * (pTarget12_ - 2*preTarget_ + pre2Target_).squaredNorm();
    double nominalPosReward = rewardCoeff.at(RewardType::NOMINALPOS) * (gc_.tail(12) - gc_init_.tail(12)).cwiseProduct(nominalJointPosWeight_).squaredNorm();
    double jointAccelReward = rewardCoeff.at(RewardType::JOINTACCEL) * (gv_.tail(12) - preJointVel_).squaredNorm();
    double motionAvoidanceReward = rewardCoeff.at(RewardType::AVOIDANCE) * (0.8 * std::pow(bodyLinearVel_[2], 2) + 0.4 * bodyAngularVel_.head(2).cwiseAbs().sum());
    double footForceReward = rewardCoeff.at(RewardType::FOOTFORCE) * footNormalForceSS_;
    double jointLimitReward = rewardCoeff.at(RewardType::JOINTLIMIT) * (float)jointLimitMetric_;

    if (standingMode_)
      nominalPosReward *= 10.0;

    positiveRwd_[0] = cmdLinear;
    positiveRwd_[1] = cmdAngular;
    positiveRwd_[2] = airTimeReward;

    negativeRwd_[0] = jointTorqueReward;
    negativeRwd_[1] = jointSpeedReward;
    negativeRwd_[2] = footSlipReward;
    negativeRwd_[3] = footClearanceReward;
    negativeRwd_[4] = orientationReward;
    negativeRwd_[5] = smoothnessReward;
    negativeRwd_[6] = smoothness2Reward;
    negativeRwd_[7] = nominalPosReward;
    negativeRwd_[8] = jointAccelReward;
    negativeRwd_[9] = motionAvoidanceReward;
    negativeRwd_[10] = footForceReward;
    negativeRwd_[11] = jointLimitReward;
  }

  double getReward() {
    return positiveRwd_.sum() * std::exp(0.2 * negativeRwd_.sum());
  }

  void updateObservation(raisim::World *world) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    raibot->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot_);
    bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);

    obDouble_ << command_, /// command 3
        rot_.e().row(0).transpose(), /// body orientation: x-axis 3
        rot_.e().row(2).transpose(), /// body orientation: z-axis 3
        bodyAngularVel_, /// body angular velocity 3
        gc_.tail(12), /// joint angles 12
        gv_.tail(12); /// joint velocity 12
  }

  void updateEstTarget() {
    Eigen::Vector4d footHeight;
    for (int i = 0; i < footPosCur_.size(); ++i)
      footHeight[i] = footPosCur_[i].e()[2];

    /// estimation
    estTarget_ << gc_[2] - groundHeight_, /// body height 1
        bodyLinearVel_, /// body linear velocity 3
        footHeight; /// foot clearance 4
  }

  void updateHyperVision() {

    /// hypervision: Asymmetric Actor-Critic
    hvDouble_ << airtimeTotal_, /// 1
        footSlippage_, /// 1
        c_f_, /// friction coefficient 1
        footNormalForce_, /// 4
        nearFootHeightMap_; /// ground height near the foot 16
  }

  const Eigen::VectorXd& getObservation() { return obDouble_; }
  const Eigen::VectorXd& getHyperVision() { return hvDouble_; }
  const Eigen::VectorXd& getEstTarget()  { return estTarget_; }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    command_ = command.cast<double>();
    if (command_.norm() < 0.3) { command_.setZero(); }
  }

  bool isTerminalState(raisim::World *world) {
    auto* raibot = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    for(auto& contact: raibot->getContacts()){
      if(std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }
    }
    return false;
  }

  const Eigen::VectorXd& getStepData() {
    stepData_.head((int)positiveRwdTag_.size()) = positiveRwd_;
    stepData_.tail((int)negativeRwdTag_.size()) = negativeRwd_;
    return stepData_;
  }

  void setSeed(int seed) { gen_.seed(seed); }
  int getObDim() const { return obDim_; }
  int getHvDim() const { return hvDim_; }
  int getEstDim() const { return estDim_; }
  int getActionDim() const { return actionDim_; }
  const std::vector<std::string>& getStepDataTag() { return stepDataTag_; }

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;

private:
  bool standingMode_;
  size_t inContact_, idx_;
  raisim::Mat<3,3> rot_;
  int gcDim_, gvDim_, nJoints_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, preTarget_, pre2Target_, preJointVel_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, estTarget_, hvDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, command_;
  std::vector<size_t> footIndices_, footFrameIndices_, footContactMask_;
  int obDim_=0, actionDim_=0, hvDim_=0, estDim_=0;
  Eigen::VectorXd stepData_, positiveRwd_, negativeRwd_;
  std::vector<std::string> stepDataTag_, positiveRwdTag_, negativeRwdTag_;
  Eigen::Vector4d airTime_, stanceTime_, footNormalForce_;
  int jointLimitMetric_;
  double airtimeTotal_, footSlippage_, footClearance_, footNormalForceSS_, groundHeight_, c_f_;
  std::vector<raisim::Vec<3>> footPosCur_, footVelCur_;
  std::vector<std::vector<raisim::Vec<3>>> footPos_, footVel_;
  std::vector<bool> contactStateCur_;
  std::vector<std::vector<bool>> contactState_;
  Eigen::VectorXd nearFootHeightMap_, nominalJointPosWeight_;

};
thread_local std::mt19937 raisim::raibotController::gen_;
thread_local std::normal_distribution<double> raisim::raibotController::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::raibotController::uniDist_(-1., 1.);
}