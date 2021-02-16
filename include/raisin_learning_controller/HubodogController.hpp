// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include <map>
#include <cstdint>
#include "BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

class HubodogController {

 public:
  enum class RewardType : int {
    LINEAR_VEL = 1,
    ANGULAR_VEL,
    BASE_MOTION,
    TORQUE,
    SMOOTHNESS1,
    SMOOTHNESS2,
    ORIENTATION,
    JOINT_VEL,
    JOINT_ACCEL,
    AIRTIME,
    SLIP
  };

  void setSeed(int seed) { gen_.seed(seed); }

  void setNumOfRewards(int numOfReward) { numOfRewards_ = numOfReward; }

  void create(raisim::World *world) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    hubodog->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// get robot data
    gcDim_ = hubodog->getGeneralizedCoordinateDim();
    gvDim_ = hubodog->getDOF();
    nJoints_ = gvDim_ - 6;
    nLegs_ = 4;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_init_from_.setZero(gcDim_); nominalJointConfig_.setZero(nJoints_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_init_from_.setZero(gvDim_); previousJointVel_.setZero(nJoints_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    footPos_.resize(4); footVel_.resize(4);
    jointPosHist_.setZero(nJoints_ * historyLength_); jointVelHist_.setZero(nJoints_ * historyLength_);
    historyTempMem_.setZero(nJoints_ * historyLength_);
    stanceTime_.setZero(nLegs_);

    /// this is nominal configuration of hubodog
    nominalJointConfig_<< 0.0, 0.52, -1.06, 0.0, 0.52, -1.06, 0.0, 0.52, -1.06, 0.0, 0.52, -1.06;
    gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, nominalJointConfig_;

    /// set pd gains
    double pGain = 50.0, dGain = 0.5;
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(pGain);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(dGain);
    hubodog->setPdGains(jointPgain_, jointDgain_);
    hubodog->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 113;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    previousAction_.setZero(actionDim_); prepreviousAction_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action & observation scaling
    actionMean_ = nominalJointConfig_;
    actionStd_.setConstant(0.2);

    /// foot scan config
//    scanConfig_.setZero(4);
//    scanConfig_ << 6, 8, 10, 12;
//    scanPoint_.resize(4, std::vector<raisim::Vec<2>>(scanConfig_.sum()));
//    heightScan_.resize(4, raisim::VecDyn(scanConfig_.sum()));

    /// indices of links that should not make contact with ground
    footIndices_.push_back(hubodog->getBodyIdx("FR_calf"));
    footIndices_.push_back(hubodog->getBodyIdx("FL_calf"));
    footIndices_.push_back(hubodog->getBodyIdx("RR_calf"));
    footIndices_.push_back(hubodog->getBodyIdx("RL_calf"));
    footFrameIndices_.push_back(hubodog->getFrameIdxByName("FR_foot_fixed"));
    footFrameIndices_.push_back(hubodog->getFrameIdxByName("FL_foot_fixed"));
    footFrameIndices_.push_back(hubodog->getFrameIdxByName("RR_foot_fixed"));
    footFrameIndices_.push_back(hubodog->getFrameIdxByName("RL_foot_fixed"));

    numOfRewards_ = 14;
    stepData_.resize(numOfRewards_);
  }

  void init(raisim::World *world) { }

  void randomizedReset(raisim::World *world, double curriculumFactor) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    /// randomize generalized coordinates
    /// x,y position
    gc_init_[0] = 0.;
    gc_init_[1] = 0.;
    /// orientation
    raisim::Mat<3,3> rotMat, yawRot, pitchRollMat;
    raisim::Vec<4> quaternion;
    raisim::Vec<3> axis = {normDist_(gen_), normDist_(gen_), normDist_(gen_)};
    axis /= axis.norm();
    raisim::angleAxisToRotMat(axis, normDist_(gen_) * 0.2, pitchRollMat);
    raisim::angleAxisToRotMat({0,0,1}, uniDist_(gen_) * M_PI, yawRot);
    raisim::matmul(pitchRollMat, yawRot, rotMat);
    raisim::rotMatToQuat(rotMat, quaternion);
    gc_init_.segment(3, 4) = quaternion.e();
    ///joint angles
    for(int i = 0 ; i < nJoints_; i++)
      gc_init_[i+7] = nominalJointConfig_[i] + 0.2 * normDist_(gen_);

    if(uniDist_(gen_) > 0.5 && hasInitial_) {
      gc_init_ = gc_init_from_;
      gv_init_ = gv_init_from_;
    }

    /// at least one foot is in contact with the terrain
    hubodog->setGeneralizedCoordinate(gc_init_);
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; /// some arbitrary high negative value
    for(auto& foot: footFrameIndices_) {
      hubodog->getFramePosition(foot, footPosition);
//      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0), footPosition(1)) - footPosition(2);
      double terrainHeightMinusFootPosition = 0.0 - footPosition(2);
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_(2) += maxNecessaryShift;

    /// randomize generalized velocities
    /// base linear velocity
    raisim::Vec<3> bodyVel_b, bodyVel_w;
    bodyVel_b[0] = 1.0 * normDist_(gen_) * (1.0 - curriculumFactor);
    bodyVel_b[1] = 0.6 * normDist_(gen_) * (1.0 - curriculumFactor);
    bodyVel_b[2] = 0.2 * normDist_(gen_) * (1.0 - curriculumFactor);
    raisim::matvecmul(rotMat, bodyVel_b, bodyVel_w);
    /// base angular velocities (just define this in the world frame since it is isometric)
    raisim::Vec<3> bodyAng_w;
    for(int i = 0; i < 3; i++) bodyAng_w[i] = 0.8 * normDist_(gen_) * (1.0 - curriculumFactor);
    /// joint velocities
    Eigen::VectorXd jointVel(12);
    for(int i = 0; i < 12; i++) jointVel[i] = 3.0 * normDist_(gen_) * (1.0 - curriculumFactor);
    /// combine
    gv_init_ << bodyVel_w.e(), bodyAng_w.e(), jointVel;

    standingMode_ = uniDist_(gen_) > 0.8;

    if(standingMode_) {
      command_.setZero();
    }
    else {
      do {
        command_ << (1 - curriculumFactor * 0.7) * 2.0 * uniDist_(gen_),
            (1 - curriculumFactor * 0.7) * 1.0 * uniDist_(gen_),
            (1 - curriculumFactor * 0.7) * 1.2 * uniDist_(gen_);
      } while(command_.norm() < 0.4);
    }

    /// set the states
    hubodog->setState(gc_init_, gv_init_);
    updateObservation(world);
    stanceTime_.setZero();
    previousAction_ << actionMean_;
    prepreviousAction_ << previousAction_;
    for(int i = 0; i < historyLength_; i++) {
      jointPosHist_.segment(nJoints_ * i, nJoints_).setZero();
      jointVelHist_.segment(nJoints_ * i, nJoints_) = gv_init_.tail(nJoints_);
    }
  }

  void reset(raisim::World *world) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, nominalJointConfig_;
    gv_init_.setZero();
    hubodog->setGeneralizedCoordinate(gc_init_);
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; /// some arbitrary high negative value
    for(auto& foot: footFrameIndices_) {
      hubodog->getFramePosition(foot, footPosition);
//      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0), footPosition(1)) - footPosition(2);
      double terrainHeightMinusFootPosition = 0.0 - footPosition(2);
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_(2) += maxNecessaryShift;
    hubodog->setState(gc_init_, gv_init_);
    updateObservation(world);
    command_ << 1.0, 0.0, 0.0;
    stanceTime_.setZero();
    previousAction_ << actionMean_;
    prepreviousAction_ << previousAction_;
    for(int i = 0; i < historyLength_; i++) {
      jointPosHist_.segment(nJoints_ * i, nJoints_).setZero();
      jointVelHist_.segment(nJoints_ * i, nJoints_) = gv_init_.tail(nJoints_);
    }
  }

  void advance(raisim::World *world, const Eigen::VectorXf& action) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    hubodog->setPdTarget(pTarget_, vTarget_);
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    command_ = command.cast<double>();
  }

  void updateHeightScan() {
//    for(int i = 0; i < nLegs_; i++) {
//      for(int k = 0; k < scanConfig_.size(); k++) {
//        for(int j = 0; j < scanConfig_[k]; j++) {
//          const double distance = 0.10 * k;
//          const double angle = 2.0 * M_PI * double(j)/scanConfig_[k];
//          scanPoint_[i][scanConfig_.head(k).sum() + j][0] = footPos_[i][0] + controlFrameX_[0] * distance * cos(angle) + controlFrameY_[0] * distance * sin(angle);
//          scanPoint_[i][scanConfig_.head(k).sum() + j][1] = footPos_[i][1] + controlFrameX_[1] * distance * cos(angle) + controlFrameY_[1] * distance * sin(angle);
//          scanPoint_[i][scanConfig_.head(k).sum() + j][0] += normDist_(gen_) * 0.004;
//          scanPoint_[i][scanConfig_.head(k).sum() + j][1] += normDist_(gen_) * 0.004;

//          heightScan_[i][scanConfig_.head(k).sum() + j] = heightMap_->getHeight(scanPoint_[i][scanConfig_.head(k).sum() + j][0], scanPoint_[i][scanConfig_.head(k).sum() + j][1]) - footPos_[i][2];
//          heightScan_[i][scanConfig_.head(k).sum() + j] += normDist_(gen_) * 0.01;
//        }
//      }
//    }
  }

  void updateHistory() {
    historyTempMem_ = jointVelHist_;
    jointVelHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointVelHist_.tail(nJoints_) = gv_.tail(nJoints_);

    historyTempMem_ = jointPosHist_;
    jointPosHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointPosHist_.tail(nJoints_) = pTarget12_ - gc_.tail(nJoints_);
  }

  void updatePreviousAction() {
    prepreviousAction_ = previousAction_;
    previousAction_ = pTarget12_;
  }

  void getReward(raisim::World *world, const std::map<RewardType, float>& rewardCoeff, double simulation_dt, double curriculumFactor) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    previousJointVel_ = gv_.tail(nJoints_);
    updateObservation(world);

    float commandTrackingReward = 0.0;
    float negativeRewardSum = 0.0;
    float positiveRewardSum = 0.0;
    /// smooth reward
    float smoothReward1 = rewardCoeff.at(RewardType::SMOOTHNESS1) * (1 - curriculumFactor * 0.7) * (pTarget12_ - previousAction_).squaredNorm();
    float smoothReward2 = rewardCoeff.at(RewardType::SMOOTHNESS2) * (1 - curriculumFactor * 0.7) * (pTarget12_ - 2 * previousAction_ + prepreviousAction_).squaredNorm();
    /// torque reward
    float torqueReward = rewardCoeff.at(RewardType::TORQUE) * (1 - curriculumFactor * 0.7) * hubodog->getGeneralizedForce().squaredNorm();
    /// joint reward
    float jointVelReward = rewardCoeff.at(RewardType::JOINT_VEL) * (1 - curriculumFactor * 0.7) * gv_.tail(nJoints_).squaredNorm();
    float jointAccelReward = rewardCoeff.at(RewardType::JOINT_ACCEL) * (1 - curriculumFactor * 0.7) * (previousJointVel_ - gv_.tail(nJoints_)).squaredNorm();
    ///command tracking reward
    float baseMotionReward = 0.0;
    float linVelReward = rewardCoeff.at(RewardType::LINEAR_VEL) * std::exp(-1.0 * (bodyLinearVel_.head(2) - command_.head(2)).squaredNorm());
    float angVelReward = rewardCoeff.at(RewardType::ANGULAR_VEL) * std::exp(-1.5 * (bodyAngularVel_(2) - command_(2)) * (bodyAngularVel_(2) - command_(2)));
    baseMotionReward -=  0.8 * bodyLinearVel_[2] * bodyLinearVel_[2];
    baseMotionReward -=  0.2 * fabs(bodyAngularVel_[0]);
    baseMotionReward -=  0.2 * fabs(bodyAngularVel_[1]);
    baseMotionReward *= rewardCoeff.at(RewardType::BASE_MOTION);
    commandTrackingReward = linVelReward + angVelReward + baseMotionReward;
    /// orientation reward
    float orientationReward = rewardCoeff.at(RewardType::ORIENTATION) * (1 - curriculumFactor * 0.7) * std::acos(rot_(8)) * std::acos(rot_(8));
    /// slip reward
    float slipReward = 0.0;
    for(int i = 0; i < nLegs_; i++)
      if(footContactState_[i]) {
        slipReward += footVel_[i].e().head(2).squaredNorm();
      }
    slipReward *= (1 - curriculumFactor * 0.9) * rewardCoeff.at(RewardType::SLIP);
    /// airtime reward
    float airtimeReward = 0.0;

    for(int i = 0; i < nLegs_; i++)
      if(footContactState_[i])
        stanceTime_(i) = std::max(0., stanceTime_(i)) + simulation_dt;
      else
        stanceTime_(i) = std::min(0., stanceTime_(i)) - simulation_dt;

    if(standingMode_) {
      for(int i = 0; i < nLegs_; i++)
        airtimeReward += std::min(stanceTime_[i], 0.4);
    }
    else {
      for(int i = 0; i < nLegs_; i++)
        if(stanceTime_(i) < 0.4 && stanceTime_(i) > 0.0)
          airtimeReward += std::min(stanceTime_[i], 0.3);
      for(int i = 0; i < nLegs_; i++)
        if(stanceTime_[i] > -0.4 && stanceTime_[i] < 0.0)
          airtimeReward += std::min(-stanceTime_[i], 0.3);
    }
    airtimeReward *= (1 - curriculumFactor * 0.8) * rewardCoeff.at(RewardType::AIRTIME);

    negativeRewardSum = smoothReward1 + smoothReward2 + torqueReward + jointVelReward + jointAccelReward + baseMotionReward + orientationReward + slipReward;
    positiveRewardSum = linVelReward + angVelReward + airtimeReward;

    stepData_[0] = commandTrackingReward;
    stepData_[1] = linVelReward;
    stepData_[2] = angVelReward;
    stepData_[3] = baseMotionReward;
    stepData_[4] = airtimeReward;
    stepData_[5] = torqueReward;
    stepData_[6] = jointVelReward;
    stepData_[7] = jointAccelReward;
    stepData_[8] = orientationReward;
    stepData_[9] = slipReward;
    stepData_[10] = smoothReward1;
    stepData_[11] = smoothReward2;
    stepData_[12] = negativeRewardSum;
    stepData_[13] = positiveRewardSum;
  }

  void updateObservation(raisim::World *world) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    hubodog->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat(0) = gc_(3); quat(1) = gc_(4); quat(2) = gc_(5); quat(3) = gc_(6);
    raisim::quatToRotMat(quat, rot_);
    bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);

    for(int i = 0; i < nLegs_; i++) {
      hubodog->getFramePosition(footFrameIndices_[i], footPos_[i]);
      hubodog->getFrameVelocity(footFrameIndices_[i], footVel_[i]);
    }

    /// height map
    controlFrameX_ = {rot_(0), rot_(1), 0.}; /// body x axis projected on the world x-y plane, expressed in the world frame
    controlFrameX_ /= controlFrameX_.norm();
    raisim::cross(zAxis_, controlFrameX_, controlFrameY_);

    /// check if the feet are in contact with the ground
    for(auto& fs: footContactState_) fs = false;
    for(auto& contact: hubodog->getContacts()) {
      if (contact.skip()) continue;
      auto it = std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex());
      size_t index = it - footIndices_.begin();
//      if (index < nLegs_ && contact.getPairObjectIndex() == heightMap_->getIndexInWorld())
//      auto ground_ = world->getObject("ground");
      if (index < nLegs_ && contact.getPairObjectIndex() == (world->getObject("floor"))->getIndexInWorld())
        footContactState_[index] = true;
    }

    if(uniDist_(gen_) > 0.99) {
      hasInitial_ = true;
      gc_init_from_ = gc_;
      gv_init_from_ = gv_;
      gc_init_from_[0] = 0.;
      gc_init_from_[1] = 0.;
    }
  }

  const Eigen::VectorXd& getObservation() {
//    updateHeightScan();

//    obDouble_ << gc_(2) - heightMap_->getHeight(gc_(0), gc_(1)), /// body height 1
    obDouble_ << gc_(2), /// body height 1
        rot_.e().row(2).transpose(), /// body orientation 3
        gc_.tail(12), /// joint angles 12
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6
        gv_.tail(12), /// joint velocity 12
        previousAction_, /// previous action 12
        prepreviousAction_, /// preprevious action 12
//        heightScan_[0].e(), heightScan_[1].e(), heightScan_[2].e(), heightScan_[3].e(), /// height scan 144
        jointPosHist_.segment((historyLength_ - 2) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 2) * nJoints_, nJoints_), /// joint History t-0.04 24
        jointPosHist_.segment((historyLength_ - 1) * nJoints_, nJoints_), jointVelHist_.segment((historyLength_ - 1) * nJoints_, nJoints_), /// joint History t-0.02 24
        stanceTime_, /// stance Time 4
        command_; /// command_ 3

    return obDouble_;
  }

  bool isTerminalState(raisim::World *world) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    /// if the contact body is not feet
    for(auto& contact: hubodog->getContacts()) {
      if (contact.skip()) continue;
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
    }

    if(acos(rot_(8)) * 180 / M_PI > 70) {
      return true;
    }

    return false;
  }

  float getMaxJointVel() {return gv_.tail(nJoints_).cwiseAbs().maxCoeff();}

  int getObDim() {return obDim_;}

  int getActionDim() {return actionDim_;}

  const Eigen::VectorXd& getStepData() {return stepData_;}

 private:
  constexpr static int historyLength_ = 3;
  int gcDim_, gvDim_, nJoints_, nLegs_;
  int obDim_ = 0, actionDim_ = 0;
  int numOfRewards_;

  bool hasInitial_ = false;
  bool standingMode_ = false;

//  raisim::HeightMap* heightMap_;
//  RandomHeightMapGenerator terrainGenerator_;
  raisim::Mat<3,3> rot_;
  raisim::Vec<3> zAxis_ = {0., 0., 1.}, controlFrameX_, controlFrameY_;
  std::vector<raisim::Vec<3>> footPos_, footVel_;

  Eigen::VectorXd jointPgain_, jointDgain_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd gc_init_from_, gv_init_from_;
  Eigen::VectorXd nominalJointConfig_, previousJointVel_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, previousAction_, prepreviousAction_;
  Eigen::VectorXd jointPosHist_, jointVelHist_, historyTempMem_;
  Eigen::VectorXd stanceTime_;
  Eigen::VectorXd stepData_;
  Eigen::Vector3d command_, bodyLinearVel_, bodyAngularVel_;

//  Eigen::VectorXi scanConfig_;
//  std::vector<raisim::VecDyn> heightScan_;
//  std::vector<std::vector<raisim::Vec<2>>> scanPoint_;

  std::vector<size_t> footIndices_;
  std::vector<size_t> footFrameIndices_;
  std::array<bool, 4> footContactState_;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};
thread_local std::mt19937 raisim::HubodogController::gen_;
thread_local std::normal_distribution<double> raisim::HubodogController::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::HubodogController::uniDist_(-1., 1.);
}