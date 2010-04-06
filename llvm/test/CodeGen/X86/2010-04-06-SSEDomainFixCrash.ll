; RUN: llc < %s -O3 -relocation-model=pic -disable-fp-elim -mcpu=nocona
;
; This test case is reduced from Bullet. It crashes SSEDomainFix.
;
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0"

%struct.CONTACT_KEY_TOKEN_COMP = type <{ i8 }>
%struct.GIM_AABB = type { %struct.btSimdScalar, %struct.btSimdScalar }
%struct.HullDesc = type { i32, i32, %struct.btSimdScalar*, i32, float, i32, i32 }
%struct.HullLibrary = type { %"struct.btAlignedObjectArray<btHullTriangle*>", %"struct.btAlignedObjectArray<btSoftBody::ePSolver::_>" }
%struct.HullResult = type { i8, i32, %"struct.btAlignedObjectArray<btVector3>", i32, i32, %"struct.btAlignedObjectArray<btSoftBody::ePSolver::_>" }
%struct.btActionInterface = type { i32 (...)** }
%"struct.btAlignedObjectArray<bool>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, i8*, i8 }
%"struct.btAlignedObjectArray<btCollisionObject*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %struct.btCollisionObject**, i8 }
%"struct.btAlignedObjectArray<btDbvt::sStkCLN>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btDbvt::sStkCLN"*, i8 }
%"struct.btAlignedObjectArray<btHullTriangle*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %struct.btHullTriangle**, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Anchor>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Anchor"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Cluster*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Cluster"**, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Face>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Face"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Joint*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Joint"**, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Link>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Link"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Material*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Material"**, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Node*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Node"**, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Node>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Node"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Note>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Note"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::RContact>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::RContact"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::SContact>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::SContact"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::Tetra>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSoftBody::Tetra"*, i8 }
%"struct.btAlignedObjectArray<btSoftBody::ePSolver::_>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, i32*, i8 }
%"struct.btAlignedObjectArray<btSparseSdf<3>::Cell*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %"struct.btSparseSdf<3>::Cell"**, i8 }
%"struct.btAlignedObjectArray<btTypedConstraint*>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %struct.btTypedConstraint**, i8 }
%"struct.btAlignedObjectArray<btVector3>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, %struct.btSimdScalar*, i8 }
%"struct.btAlignedObjectArray<float>" = type { %struct.CONTACT_KEY_TOKEN_COMP, i32, i32, float*, i8 }
%struct.btBroadphaseProxy = type { i8*, i16, i16, i8*, i32, %struct.btSimdScalar, %struct.btSimdScalar }
%struct.btCollisionObject = type { i32 (...)**, %struct.btTransform, %struct.btTransform, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, i8, float, %struct.btBroadphaseProxy*, %struct.btCollisionShape*, %struct.btCollisionShape*, i32, i32, i32, i32, float, float, float, i8*, i32, float, float, float, i8, [7 x i8] }
%struct.btCollisionShape = type { i32 (...)**, i32, i8* }
%struct.btDbvt = type { %struct.btDbvtNode*, %struct.btDbvtNode*, i32, i32, i32, %"struct.btAlignedObjectArray<btDbvt::sStkCLN>" }
%"struct.btDbvt::sStkCLN" = type { %struct.btDbvtNode*, %struct.btDbvtNode* }
%struct.btDbvtNode = type { %struct.GIM_AABB, %struct.btDbvtNode*, %"union.btDbvtNode::$_12" }
%"struct.btHashKey<btTriIndex>" = type { i32 }
%struct.btHullTriangle = type { %struct.int3, %struct.int3, i32, i32, float }
%struct.btMatrix3x3 = type { [3 x %struct.btSimdScalar] }
%"struct.btRaycastVehicle::btVehicleTuning" = type { float, float, float, float, float }
%struct.btRigidBody = type { %struct.btCollisionObject, %struct.btMatrix3x3, %struct.btSimdScalar, %struct.btSimdScalar, float, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, float, float, i8, float, float, float, float, float, float, %struct.btActionInterface*, %"struct.btAlignedObjectArray<btTypedConstraint*>", i32, i32, i32 }
%struct.btSimdScalar = type { %"union.btSimdScalar::$_13" }
%struct.btSoftBody = type { [268 x i8], %"struct.btAlignedObjectArray<btCollisionObject*>", %"struct.btSoftBody::Config", %"struct.btRaycastVehicle::btVehicleTuning", %"struct.btSoftBody::Pose", i8*, %struct.btSoftBodyWorldInfo*, %"struct.btAlignedObjectArray<btSoftBody::Note>", %"struct.btAlignedObjectArray<btSoftBody::Node>", %"struct.btAlignedObjectArray<btSoftBody::Link>", %"struct.btAlignedObjectArray<btSoftBody::Face>", %"struct.btAlignedObjectArray<btSoftBody::Tetra>", %"struct.btAlignedObjectArray<btSoftBody::Anchor>", %"struct.btAlignedObjectArray<btSoftBody::RContact>", %"struct.btAlignedObjectArray<btSoftBody::SContact>", %"struct.btAlignedObjectArray<btSoftBody::Joint*>", %"struct.btAlignedObjectArray<btSoftBody::Material*>", float, [2 x %struct.btSimdScalar], i8, %struct.btDbvt, %struct.btDbvt, %struct.btDbvt, %"struct.btAlignedObjectArray<btSoftBody::Cluster*>", %"struct.btAlignedObjectArray<bool>", %struct.btTransform, %"struct.btAlignedObjectArray<btSoftBody::ePSolver::_>" }
%"struct.btSoftBody::Anchor" = type { %"struct.btSoftBody::Node"*, %struct.btSimdScalar, %struct.btRigidBody*, %struct.btMatrix3x3, %struct.btSimdScalar, float }
%"struct.btSoftBody::Body" = type { %"struct.btSoftBody::Cluster"*, %struct.btRigidBody*, %struct.btCollisionObject* }
%"struct.btSoftBody::Cluster" = type { %"struct.btAlignedObjectArray<btSoftBody::Node*>", %"struct.btAlignedObjectArray<float>", %"struct.btAlignedObjectArray<btVector3>", %struct.btTransform, float, float, %struct.btMatrix3x3, %struct.btMatrix3x3, %struct.btSimdScalar, [2 x %struct.btSimdScalar], [2 x %struct.btSimdScalar], i32, i32, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btDbvtNode*, float, float, float, float, float, float, i8, i8, i32 }
%"struct.btSoftBody::Config" = type { i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, i32, i32, i32, i32, i32, %"struct.btAlignedObjectArray<btSoftBody::ePSolver::_>", %"struct.btAlignedObjectArray<btSoftBody::ePSolver::_>", %"struct.btAlignedObjectArray<btSoftBody::ePSolver::_>" }
%"struct.btSoftBody::Element" = type { i8* }
%"struct.btSoftBody::Face" = type { %"struct.btSoftBody::Feature", [3 x %"struct.btSoftBody::Node"*], %struct.btSimdScalar, float, %struct.btDbvtNode* }
%"struct.btSoftBody::Feature" = type { %"struct.btSoftBody::Element", %"struct.btSoftBody::Material"* }
%"struct.btSoftBody::Joint" = type { i32 (...)**, [2 x %"struct.btSoftBody::Body"], [2 x %struct.btSimdScalar], float, float, float, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btMatrix3x3, i8 }
%"struct.btSoftBody::Link" = type { %"struct.btSoftBody::Feature", [2 x %"struct.btSoftBody::Node"*], float, i8, float, float, float, %struct.btSimdScalar }
%"struct.btSoftBody::Material" = type { %"struct.btSoftBody::Element", float, float, float, i32 }
%"struct.btSoftBody::Node" = type { %"struct.btSoftBody::Feature", %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar, float, float, %struct.btDbvtNode*, i8 }
%"struct.btSoftBody::Note" = type { %"struct.btSoftBody::Element", i8*, %struct.btSimdScalar, i32, [4 x %"struct.btSoftBody::Node"*], [4 x float] }
%"struct.btSoftBody::Pose" = type { i8, i8, float, %"struct.btAlignedObjectArray<btVector3>", %"struct.btAlignedObjectArray<float>", %struct.btSimdScalar, %struct.btMatrix3x3, %struct.btMatrix3x3, %struct.btMatrix3x3 }
%"struct.btSoftBody::RContact" = type { %"struct.btSoftBody::sCti", %"struct.btSoftBody::Node"*, %struct.btMatrix3x3, %struct.btSimdScalar, float, float, float }
%"struct.btSoftBody::SContact" = type { %"struct.btSoftBody::Node"*, %"struct.btSoftBody::Face"*, %struct.btSimdScalar, %struct.btSimdScalar, float, float, [2 x float] }
%"struct.btSoftBody::Tetra" = type { %"struct.btSoftBody::Feature", [4 x %"struct.btSoftBody::Node"*], float, %struct.btDbvtNode*, [4 x %struct.btSimdScalar], float, float }
%"struct.btSoftBody::sCti" = type { %struct.btCollisionObject*, %struct.btSimdScalar, float }
%struct.btSoftBodyWorldInfo = type { float, float, float, %struct.btSimdScalar, %struct.btActionInterface*, %struct.btActionInterface*, %struct.btSimdScalar, %"struct.btSparseSdf<3>" }
%"struct.btSparseSdf<3>" = type { %"struct.btAlignedObjectArray<btSparseSdf<3>::Cell*>", float, i32, i32, i32, i32 }
%"struct.btSparseSdf<3>::Cell" = type { [4 x [4 x [4 x float]]], [3 x i32], i32, i32, %struct.btCollisionShape*, %"struct.btSparseSdf<3>::Cell"* }
%struct.btTransform = type { %struct.btMatrix3x3, %struct.btSimdScalar }
%struct.btTypedConstraint = type { i32 (...)**, %"struct.btHashKey<btTriIndex>", i32, i32, i8, %struct.btRigidBody*, %struct.btRigidBody*, float, float, %struct.btSimdScalar, %struct.btSimdScalar, %struct.btSimdScalar }
%struct.int3 = type { i32, i32, i32 }
%"union.btDbvtNode::$_12" = type { [2 x %struct.btDbvtNode*] }
%"union.btSimdScalar::$_13" = type { <4 x float> }

declare i32 @_ZN11HullLibrary16CreateConvexHullERK8HullDescR10HullResult(%struct.HullLibrary*, %struct.HullDesc* nocapture, %struct.HullResult* nocapture) ssp align 2

define void @_ZN17btSoftBodyHelpers4DrawEP10btSoftBodyP12btIDebugDrawi(%struct.btSoftBody* %psb, %struct.btActionInterface* %idraw, i32 %drawflags) ssp align 2 {
entry:
  br i1 undef, label %bb92, label %bb58

bb58:                                             ; preds = %entry
  %0 = invoke i32 @_ZN11HullLibrary16CreateConvexHullERK8HullDescR10HullResult(%struct.HullLibrary* undef, %struct.HullDesc* undef, %struct.HullResult* undef)
          to label %invcont64 unwind label %lpad159 ; <i32> [#uses=0]

invcont64:                                        ; preds = %bb58
  br i1 undef, label %invcont65, label %bb.i.i

bb.i.i:                                           ; preds = %invcont64
  %1 = load <4 x float>* undef, align 16          ; <<4 x float>> [#uses=5]
  br i1 undef, label %bb.nph.i.i, label %invcont65

bb.nph.i.i:                                       ; preds = %bb.i.i
  %tmp22.i.i = bitcast <4 x float> %1 to i128     ; <i128> [#uses=1]
  %tmp23.i.i = trunc i128 %tmp22.i.i to i32       ; <i32> [#uses=1]
  %2 = bitcast i32 %tmp23.i.i to float            ; <float> [#uses=1]
  %tmp6.i = extractelement <4 x float> %1, i32 1  ; <float> [#uses=1]
  %tmp2.i = extractelement <4 x float> %1, i32 2  ; <float> [#uses=1]
  br label %bb1.i.i

bb1.i.i:                                          ; preds = %bb1.i.i, %bb.nph.i.i
  %.tmp6.0.i.i = phi float [ %tmp2.i, %bb.nph.i.i ], [ %5, %bb1.i.i ] ; <float> [#uses=1]
  %.tmp5.0.i.i = phi float [ %tmp6.i, %bb.nph.i.i ], [ %4, %bb1.i.i ] ; <float> [#uses=1]
  %.tmp.0.i.i = phi float [ %2, %bb.nph.i.i ], [ %3, %bb1.i.i ] ; <float> [#uses=1]
  %3 = fadd float %.tmp.0.i.i, undef              ; <float> [#uses=2]
  %4 = fadd float %.tmp5.0.i.i, undef             ; <float> [#uses=2]
  %5 = fadd float %.tmp6.0.i.i, undef             ; <float> [#uses=2]
  br i1 undef, label %bb2.return.loopexit_crit_edge.i.i, label %bb1.i.i

bb2.return.loopexit_crit_edge.i.i:                ; preds = %bb1.i.i
  %tmp8.i = insertelement <4 x float> %1, float %3, i32 0 ; <<4 x float>> [#uses=1]
  %tmp4.i = insertelement <4 x float> %tmp8.i, float %4, i32 1 ; <<4 x float>> [#uses=1]
  %tmp.i = insertelement <4 x float> %tmp4.i, float %5, i32 2 ; <<4 x float>> [#uses=1]
  br label %invcont65

invcont65:                                        ; preds = %bb2.return.loopexit_crit_edge.i.i, %bb.i.i, %invcont64
  %.0.i = phi <4 x float> [ %tmp.i, %bb2.return.loopexit_crit_edge.i.i ], [ undef, %invcont64 ], [ %1, %bb.i.i ] ; <<4 x float>> [#uses=1]
  %tmp15.i = extractelement <4 x float> %.0.i, i32 2 ; <float> [#uses=1]
  %6 = fmul float %tmp15.i, undef                 ; <float> [#uses=1]
  br label %bb.i265

bb.i265:                                          ; preds = %bb.i265, %invcont65
  %7 = fsub float 0.000000e+00, %6                ; <float> [#uses=1]
  store float %7, float* undef, align 4
  br label %bb.i265

bb92:                                             ; preds = %entry
  unreachable

lpad159:                                          ; preds = %bb58
  unreachable
}
