; Test VectorType handling by SCCP. 
; SCCP ignores VectorTypes until PR 1034 is fixed
;
; RUN: llvm-upgrade < %s | llvm-as | opt -sccp 
; END.
target datalayout = "E-p:32:32"
target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8"
	%struct.GLDAlphaTest = type { float, ushort, ubyte, ubyte }
	%struct.GLDArrayRange = type { ubyte, ubyte, ubyte, ubyte }
	%struct.GLDBlendMode = type { ushort, ushort, ushort, ushort, %struct.GLTColor4, ushort, ushort, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDBufferRec = type opaque
	%struct.GLDBufferstate = type { %struct.GLTDimensions, %struct.GLTDimensions, %struct.GLTFixedColor4, %struct.GLTFixedColor4, ubyte, ubyte, ubyte, ubyte, [2 x %struct.GLSBuffer], [4 x %struct.GLSBuffer], %struct.GLSBuffer, %struct.GLSBuffer, %struct.GLSBuffer, [4 x %struct.GLSBuffer*], %struct.GLSBuffer*, %struct.GLSBuffer*, %struct.GLSBuffer*, ubyte, ubyte }
	%struct.GLDClearColor = type { double, %struct.GLTColor4, %struct.GLTColor4, float, int }
	%struct.GLDClipPlane = type { uint, [6 x %struct.GLTColor4] }
	%struct.GLDColorBuffer = type { ushort, ushort, [4 x ushort] }
	%struct.GLDColorMatrix = type { [16 x float]*, %struct.GLDImagingColorScale }
	%struct.GLDContextRec = type { float, float, float, float, float, float, float, float, %struct.GLTColor4, %struct.GLTColor4, %struct.GLVMFPContext, %struct.GLDTextureMachine, %struct.GLGProcessor, %struct._GLVMConstants*, void (%struct.GLDContextRec*, int, int, %struct.GLVMFragmentAttribRec*, %struct.GLVMFragmentAttribRec*, int)*, %struct._GLVMFunction*, void (%struct.GLDContextRec*, %struct.GLDVertex*)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, %struct.GLDVertex*)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, %struct.GLDVertex*, %struct.GLDVertex*)*, %struct._GLVMFunction*, %struct._GLVMFunction*, %struct._GLVMFunction*, uint, uint, uint, float, float, float, uint, %struct.GLSDrawable, %struct.GLDRect, %struct.GLDFormat, %struct.GLDBufferstate, %struct.GLDSharedRec*, %struct.GLDState*, %struct.GLDPluginState*, %struct.GLTDimensions, %struct.GLTColor4*, %struct.GLTColor4*, %struct.GLVMFragmentAttribRec*, %struct.GLVMFragmentAttribRec*, %struct.GLVMFragmentAttribRec*, %struct.GLDPipelineProgramRec*, %struct.GLDStateProgramRec, %struct.GLVMTextures, { [4 x sbyte*], sbyte*, sbyte* }, [64 x float], %struct.GLDStippleData, ushort, ubyte, ubyte, uint, %struct.GLDFramebufferRec*, ubyte, %struct.GLDQueryRec*, %struct.GLDQueryRec* }
	%struct.GLDConvolution = type { %struct.GLTColor4, %struct.GLDImagingColorScale, ushort, ushort, float*, int, int }
	%struct.GLDDepthTest = type { ushort, ushort, ubyte, ubyte, ubyte, ubyte, double, double }
	%struct.GLDFogMode = type { %struct.GLTColor4, float, float, float, float, float, ushort, ushort, ushort, ubyte, ubyte }
	%struct.GLDFormat = type { int, int, int, int, int, int, uint, uint, ubyte, ubyte, ubyte, ubyte, int, int, int }
	%struct.GLDFramebufferAttachment = type { uint, uint, uint, int, uint, uint }
	%struct.GLDFramebufferData = type { [6 x %struct.GLDFramebufferAttachment], [4 x ushort], ushort, ushort, ushort, ushort, uint }
	%struct.GLDFramebufferRec = type { %struct.GLDFramebufferData*, %struct.GLDPluginFramebufferData*, %struct.GLDPixelFormat }
	%struct.GLDHintMode = type { ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort }
	%struct.GLDHistogram = type { %struct.GLTFixedColor4*, int, ushort, ubyte, ubyte }
	%struct.GLDImagingColorScale = type { { float, float }, { float, float }, { float, float }, { float, float } }
	%struct.GLDImagingSubset = type { %struct.GLDConvolution, %struct.GLDConvolution, %struct.GLDConvolution, %struct.GLDColorMatrix, %struct.GLDMinmax, %struct.GLDHistogram, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, uint }
	%struct.GLDLight = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTCoord3, float, float, float, float, float, %struct.GLTCoord3, float, float, float, float, float }
	%struct.GLDLightModel = type { %struct.GLTColor4, [8 x %struct.GLDLight], [2 x %struct.GLDMaterial], uint, ushort, ushort, ushort, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDLightProduct = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4 }
	%struct.GLDLineMode = type { float, int, ushort, ushort, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDLogicOp = type { ushort, ubyte, ubyte }
	%struct.GLDMaskMode = type { uint, [3 x uint], ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDMaterial = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, float, float, float, float, [8 x %struct.GLDLightProduct], %struct.GLTColor4, [6 x int], [2 x int] }
	%struct.GLDMinmax = type { %struct.GLDMinmaxTable*, ushort, ubyte, ubyte }
	%struct.GLDMinmaxTable = type { %struct.GLTColor4, %struct.GLTColor4 }
	%struct.GLDMipmaplevel = type { [4 x uint], [4 x float], [4 x uint], [4 x uint], [4 x float], [4 x uint], [3 x uint], uint, float*, float*, float*, uint, uint, sbyte*, short, ushort, ushort, short }
	%struct.GLDMultisample = type { float, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDPipelineProgramData = type { ushort, ushort, uint, %struct._PPStreamToken*, ulong, %struct.GLDShaderSourceData*, %struct.GLTColor4*, uint }
	%struct.GLDPipelineProgramRec = type { %struct.GLDPipelineProgramData*, %struct._PPStreamToken*, %struct._PPStreamToken*, %struct._GLVMFunction*, uint, uint, uint }
	%struct.GLDPipelineProgramState = type { ubyte, ubyte, ubyte, ubyte, %struct.GLTColor4* }
	%struct.GLDPixelFormat = type { ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDPixelMap = type { int*, float*, float*, float*, float*, float*, float*, float*, float*, int*, int, int, int, int, int, int, int, int, int, int }
	%struct.GLDPixelMode = type { float, float, %struct.GLDPixelStore, %struct.GLDPixelTransfer, %struct.GLDPixelMap, %struct.GLDImagingSubset, uint, uint }
	%struct.GLDPixelPack = type { int, int, int, int, int, int, int, int, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDPixelStore = type { %struct.GLDPixelPack, %struct.GLDPixelPack }
	%struct.GLDPixelTransfer = type { float, float, float, float, float, float, float, float, float, float, int, int, float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.GLDPluginFramebufferData = type { [6 x %struct.GLDTextureRec*], uint, uint }
	%struct.GLDPluginProgramData = type { [3 x %struct.GLDPipelineProgramRec*], %struct.GLDBufferRec**, uint }
	%struct.GLDPluginState = type { [16 x [5 x %struct.GLDTextureRec*]], [3 x %struct.GLDTextureRec*], [16 x %struct.GLDTextureRec*], [3 x %struct.GLDPipelineProgramRec*], %struct.GLDProgramRec*, %struct.GLDVertexArrayRec*, [16 x %struct.GLDBufferRec*], %struct.GLDFramebufferRec*, %struct.GLDFramebufferRec* }
	%struct.GLDPointMode = type { float, float, float, float, %struct.GLTCoord3, float, ubyte, ubyte, ubyte, ubyte, ushort, ushort, uint, ushort, ushort }
	%struct.GLDPolygonMode = type { [128 x ubyte], float, float, ushort, ushort, ushort, ushort, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDProgramData = type { uint, [16 x int], int, int, uint, int }
	%struct.GLDProgramRec = type { %struct.GLDProgramData*, %struct.GLDPluginProgramData*, uint }
	%struct.GLDQueryRec = type { uint, uint, %struct.GLDQueryRec* }
	%struct.GLDRect = type { int, int, int, int, int, int }
	%struct.GLDRegisterCombiners = type { ubyte, ubyte, ubyte, ubyte, int, [2 x %struct.GLTColor4], [8 x %struct.GLDRegisterCombinersPerStageState], %struct.GLDRegisterCombinersFinalStageState }
	%struct.GLDRegisterCombinersFinalStageState = type { ubyte, ubyte, ubyte, ubyte, [7 x %struct.GLDRegisterCombinersPerVariableState] }
	%struct.GLDRegisterCombinersPerPortionState = type { [4 x %struct.GLDRegisterCombinersPerVariableState], ubyte, ubyte, ubyte, ubyte, ushort, ushort, ushort, ushort, ushort, ushort }
	%struct.GLDRegisterCombinersPerStageState = type { [2 x %struct.GLDRegisterCombinersPerPortionState], [2 x %struct.GLTColor4] }
	%struct.GLDRegisterCombinersPerVariableState = type { ushort, ushort, ushort, ushort }
	%struct.GLDScissorTest = type { %struct.GLTFixedColor4, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDShaderSourceData = type { uint, uint, sbyte*, int*, uint, uint, sbyte*, int*, sbyte* }
	%struct.GLDSharedRec = type opaque
	%struct.GLDState = type { short, short, uint, uint, uint, [256 x %struct.GLTColor4], [128 x %struct.GLTColor4], %struct.GLDViewport, %struct.GLDTransform, %struct.GLDLightModel, uint*, int, int, int, %struct.GLDAlphaTest, %struct.GLDBlendMode, %struct.GLDClearColor, %struct.GLDColorBuffer, %struct.GLDDepthTest, %struct.GLDArrayRange, %struct.GLDFogMode, %struct.GLDHintMode, %struct.GLDLineMode, %struct.GLDLogicOp, %struct.GLDMaskMode, %struct.GLDPixelMode, %struct.GLDPointMode, %struct.GLDPolygonMode, %struct.GLDScissorTest, uint, %struct.GLDStencilTest, [16 x %struct.GLDTextureMode], %struct.GLDArrayRange, [8 x %struct.GLDTextureCoordGen], %struct.GLDClipPlane, %struct.GLDMultisample, %struct.GLDRegisterCombiners, %struct.GLDArrayRange, %struct.GLDArrayRange, [3 x %struct.GLDPipelineProgramState], %struct.GLDTransformFeedback }
	%struct.GLDStateProgramRec = type { %struct.GLDPipelineProgramData*, %struct.GLDPipelineProgramRec* }
	%struct.GLDStencilTest = type { [3 x { uint, int, ushort, ushort, ushort, ushort }], uint, [4 x ubyte] }
	%struct.GLDStippleData = type { uint, ushort, ushort, [32 x [32 x ubyte]] }
	%struct.GLDTextureCoordGen = type { { ushort, ushort, %struct.GLTColor4, %struct.GLTColor4 }, { ushort, ushort, %struct.GLTColor4, %struct.GLTColor4 }, { ushort, ushort, %struct.GLTColor4, %struct.GLTColor4 }, { ushort, ushort, %struct.GLTColor4, %struct.GLTColor4 }, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDTextureGeomState = type { ushort, ushort, ushort, ushort, ushort, ubyte, ubyte, ushort, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, [6 x ushort], [6 x ushort] }
	%struct.GLDTextureLevel = type { uint, uint, ushort, ushort, ushort, ubyte, ubyte, ushort, ushort, ushort, ushort, ubyte* }
	%struct.GLDTextureMachine = type { [8 x %struct.GLDTextureRec*], %struct.GLDTextureRec*, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDTextureMode = type { %struct.GLTColor4, uint, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, float, float, float, ushort, ushort, ushort, ushort, ushort, ushort, [4 x ushort], ubyte, ubyte, ubyte, ubyte, [3 x float], [4 x float], float, float }
	%struct.GLDTextureParamState = type { ushort, ushort, ushort, ushort, ushort, ushort, %struct.GLTColor4, float, float, float, float, short, short, ushort, ushort, float, ushort, ubyte, ubyte, int, sbyte* }
	%struct.GLDTextureRec = type { %struct.GLDTextureState*, int, [2 x float], float, uint, float, float, float, float, float, float, %struct.GLDMipmaplevel*, %struct.GLDMipmaplevel*, int, int, uint, uint, uint, uint, %struct.GLDTextureParamState, uint, [2 x %struct._PPStreamToken] }
	%struct.GLDTextureState = type { ushort, ushort, ushort, float, uint, ushort, %struct.GLISWRSurface*, ubyte, ubyte, ubyte, ubyte, %struct.GLDTextureParamState, %struct.GLDTextureGeomState, %struct.GLDTextureLevel, [6 x [15 x %struct.GLDTextureLevel]] }
	%struct.GLDTransform = type { [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, int, float, ushort, ushort, ubyte, ubyte, ubyte, ubyte }
	%struct.GLDTransformFeedback = type { ubyte, ubyte, ubyte, [16 x uint], [16 x uint] }
	%struct.GLDVertex = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTCoord3, float, %struct.GLTColor4, float, float, float, ubyte, ubyte, ubyte, ubyte, [4 x float], [2 x %struct.GLDMaterial*], uint, uint, [8 x %struct.GLTColor4] }
	%struct.GLDVertexArrayRec = type opaque
	%struct.GLDViewport = type { float, float, float, float, float, float, float, float, double, double, int, int, int, int, float, float, float, float }
	%struct.GLGColorTable = type { uint, uint, int, sbyte* }
	%struct.GLGOperation = type { sbyte*, sbyte*, int, uint, uint, int, uint, uint, uint, uint, uint, uint, uint, float, float, %struct.GLGColorTable, %struct.GLGColorTable, %struct.GLGColorTable }
	%struct.GLGProcessor = type { void (%struct.GLDPixelMode*, %struct.GLGOperation*, %struct._GLGFunctionKey*)*, %struct._GLVMFunction*, %struct._GLGFunctionKey* }
	%struct.GLISWRSurface = type { int, int, int, int, int, int, int, int, int, int, ubyte*, ubyte*, ubyte*, [4 x ubyte*], uint }
	%struct.GLIWindow = type { uint, uint, uint }
	%struct.GLSBuffer = type { sbyte* }
	%struct.GLSDrawable = type { %struct.GLSWindowRec* }
	%struct.GLSWindowRec = type { %struct.GLTDimensions, %struct.GLTDimensions, uint, uint, %struct.GLSDrawable, [2 x ubyte*], ubyte*, ubyte*, ubyte*, [4 x ubyte*], uint, uint, uint, uint, [4 x uint], ushort, ushort, ushort, %struct.GLIWindow, uint, uint, sbyte*, ubyte* }
	%struct.GLTColor4 = type { float, float, float, float }
	%struct.GLTCoord3 = type { float, float, float }
	%struct.GLTDimensions = type { int, int }
	%struct.GLTFixedColor4 = type { int, int, int, int }
	%struct.GLVMFPContext = type { float, uint, uint, uint }
	%struct.GLVMFragmentAttribRec = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, [8 x <4 x float>] }
	%struct.GLVMTextures = type { [8 x %struct.GLDTextureRec*] }
	%struct._GLGFunctionKey = type opaque
	%struct._GLVMConstants = type opaque
	%struct._GLVMFunction = type opaque
	%struct._PPStreamToken = type { { ushort, ubyte, ubyte, uint } }

implementation   ; Functions:

void %gldLLVMVecPointRender(%struct.GLDContextRec* %ctx) {
entry:
	%tmp.uip = getelementptr %struct.GLDContextRec* %ctx, int 0, uint 22		; <uint*> [#uses=1]
	%tmp = load uint* %tmp.uip		; <uint> [#uses=3]
	%tmp91 = lshr uint %tmp, ubyte 5		; <uint> [#uses=1]
	%tmp92 = trunc uint %tmp91 to bool		; <bool> [#uses=1]
	br bool %tmp92, label %cond_true93, label %cond_next116

cond_true93:		; preds = %entry
	%tmp = getelementptr %struct.GLDContextRec* %ctx, int 0, uint 31, uint 14		; <int*> [#uses=1]
	%tmp95 = load int* %tmp		; <int> [#uses=1]
	%tmp95 = sitofp int %tmp95 to float		; <float> [#uses=1]
	%tmp108 = mul float undef, %tmp95		; <float> [#uses=1]
	br label %cond_next116

cond_next116:		; preds = %cond_true93, %entry
	%point_size.2 = phi float [ %tmp108, %cond_true93 ], [ undef, %entry ]		; <float> [#uses=2]
	%tmp457 = setlt float %point_size.2, 1.000000e+00		; <bool> [#uses=1]
	%tmp460 = lshr uint %tmp, ubyte 6		; <uint> [#uses=1]
	%tmp461 = trunc uint %tmp460 to bool		; <bool> [#uses=1]
	br bool %tmp457, label %cond_true458, label %cond_next484

cond_true458:		; preds = %cond_next116
	br bool %tmp461, label %cond_true462, label %cond_next487

cond_true462:		; preds = %cond_true458
	%tmp26 = bitcast uint %tmp to int		; <int> [#uses=1]
	%tmp465 = and int %tmp26, 128		; <int> [#uses=1]
	%tmp466 = seteq int %tmp465, 0		; <bool> [#uses=1]
	br bool %tmp466, label %cond_true467, label %cond_next487

cond_true467:		; preds = %cond_true462
	ret void

cond_next484:		; preds = %cond_next116
	%tmp486 = mul float %point_size.2, 5.000000e-01		; <float> [#uses=1]
	br label %cond_next487

cond_next487:		; preds = %cond_next484, %cond_true462, %cond_true458
	%radius.0 = phi float [ %tmp486, %cond_next484 ], [ 5.000000e-01, %cond_true458 ], [ 5.000000e-01, %cond_true462 ]		; <float> [#uses=2]
	%tmp494 = insertelement <4 x float> zeroinitializer, float %radius.0, uint 2		; <<4 x float>> [#uses=1]
	%tmp495 = insertelement <4 x float> %tmp494, float %radius.0, uint 3		; <<4 x float>> [#uses=0]
	ret void
}
