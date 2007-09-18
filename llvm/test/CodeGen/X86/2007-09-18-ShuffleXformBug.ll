; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep 170

	%struct.XXAlphaTest = type { float, i16, i8, i8 }
	%struct.XXArrayRange = type { i8, i8, i8, i8 }
	%struct.XXBlendMode = type { i16, i16, i16, i16, %struct.GGC4, i16, i16, i8, i8, i8, i8 }
	%struct.XXBufferData = type { i8*, i32, i32, i16, i16, i8, i8, i8, i8 }
	%struct.XXBufferRec = type opaque
	%struct.XXBufferstate = type { %struct.GLGXfKey, %struct.GLGXfKey, %struct.XXFramebufferAttachment, %struct.XXFramebufferAttachment, i8, i8, i8, i8, [2 x %struct.GLSBuffer], [4 x %struct.GLSBuffer], %struct.GLSBuffer, %struct.GLSBuffer, %struct.GLSBuffer, [4 x %struct.GLSBuffer*], %struct.GLSBuffer*, %struct.GLSBuffer*, %struct.GLSBuffer*, i8, i8 }
	%struct.XXClearC = type { double, %struct.GGC4, %struct.GGC4, float, i32 }
	%struct.XXClipPlane = type { i32, [6 x %struct.GGC4] }
	%struct.XXCBuffer = type { i16, i16, [8 x i16] }
	%struct.XXCMatrix = type { [16 x float]*, %struct.XXImagingCScale }
	%struct.XXConfig = type { i32, float, %struct.GLGXfKey, %struct.GLGXfKey, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, %struct.XXPixelFormat, %struct.XXPointLineLimits, %struct.XXPointLineLimits, %struct.XXRenderFeatures, %struct.XXVArrayTypes, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.XXTextureLimits, [3 x %struct.XXPipelineProgramLimits], %struct.XXFragmentProgramLimits, %struct.XXVProgramLimits, %struct.XXGeometryShaderLimits, %struct.XXProgramLimits, %struct.XXXfFeedbackLimits, %struct.XXVDescriptor*, %struct.XXVDescriptor*, [3 x i32] }
	%struct.XXContextRec = type { float, float, float, float, float, float, float, float, %struct.GGC4, %struct.GGC4, %struct.LLFPContext, [16 x [2 x %struct.PPStreamToken]], %struct.GLGProcessor, %struct._LLConstants*, void (%struct.XXContextRec*, i32, i32, %struct.LLFragmentAttrib*, %struct.LLFragmentAttrib*, i32)*, %struct._LLFunction*, %struct.PPStreamToken*, void (%struct.XXContextRec*, %struct.XXV*)*, void (%struct.XXContextRec*, %struct.XXV*, %struct.XXV*)*, void (%struct.XXContextRec*, %struct.XXV*, %struct.XXV*, %struct.XXV*)*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, [2 x i32], [1 x i32], [1 x i32], float, float, float, i32, i32, %struct.GLSDrawable, %struct.XXFramebufferRec*, %struct.XXRect, %struct.XXFormat, %struct.XXFormat, %struct.XXConfig*, %struct.XXBufferstate, %struct.XXSharedRec*, %struct.XXState*, %struct.XXPluginState*, %struct.GGC4*, %struct.GGC4*, %struct.LLFragmentAttrib*, %struct.LLFragmentAttrib*, %struct.LLFragmentAttrib*, %struct.XXProgramRec*, %struct.XXPipelineProgramRec*, %struct.LLTextures, { [4 x i8*], i8*, i8*, i8* }, %struct.XXStippleData, i16, i8, i8, i32, i32, %struct.XXQueryRec*, %struct.XXQueryRec*, %struct.XXFallback }
	%struct.XXConvolution = type { %struct.GGC4, %struct.XXImagingCScale, i16, i16, float*, i32, i32 }
	%struct.XXCurrent16A = type { [8 x %struct.GGC4], [16 x %struct.GGC4], %struct.GGC4, %struct.XXPointLineLimits, float, %struct.XXPointLineLimits, float, [4 x float], %struct.XXPointLineLimits, float, float, float, float, i8, i8, i8, i8 }
	%struct.XXDepthTest = type { i16, i16, i8, i8, i8, i8, double, double }
	%struct.XXDispatch = type { i8 (i32*, i32*, i32*, i32*) zeroext *, i32 (%struct.GGPixelFormat**, i32*)*, i32 (%struct.GGPixelFormat*)*, i32 (%struct.GGRendererInfo*, i32)*, i32 (%struct.XXSharedRec**)*, i32 (%struct.XXSharedRec*)*, i32 (%struct.XXContextRec**, %struct.GGPixelFormat*, %struct.XXSharedRec*, %struct.XXContextRec*, %struct.XXConfig*, %struct.XXState*, %struct.XXPluginState*)*, i32 (%struct.XXContextRec*)*, void (%struct.XXContextRec*)*, i32 (%struct.XXContextRec*, i32, i8*, i32)*, i32 (%struct.XXContextRec*, i32, i32*)*, i32 (%struct.XXContextRec*, i32, i32*)*, i32 (%struct.XXContextRec*, %struct.XXRenderDispatch*, %struct.XXViewportConfig*)*, i32 (%struct.XXContextRec*, %struct.XXRenderDispatch*, i32*)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec**, %struct.XXTextureState*)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i32)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i32, i32, i32, i32, i32, i32)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i32, i32*)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i8*)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*)*, i8 (%struct.XXContextRec*, %struct.XXTextureRec*) zeroext *, void (%struct.XXContextRec*, %struct.XXTextureRec*)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*)*, i8* (%struct.XXContextRec*, i32)*, i32 (%struct.XXContextRec*)*, i8* (%struct.XXContextRec*, i32, i32*)*, void (%struct.XXContextRec*, i8*, i32)*, void (%struct.XXContextRec*, i8*)*, i32 (%struct.XXContextRec*, %struct.XXPipelineProgramRec**, %struct.XXPipelineProgramData*)*, i32 (%struct.XXContextRec*, %struct.XXPipelineProgramRec*, i32)*, i32 (%struct.XXContextRec*, %struct.XXPipelineProgramRec*, i32, i32*)*, i32 (%struct.XXContextRec*, %struct.XXPipelineProgramRec*)*, i32 (%struct.XXContextRec*, %struct.XXProgramRec**, %struct.XXProgramData*, %struct.XXPluginProgramData*)*, i32 (%struct.XXContextRec*, %struct.XXProgramRec*)*, i32 (%struct.XXContextRec*, %struct.XXVArrayRec**, %struct.XXVArrayData*, %struct.XXPluginVArrayData*)*, i32 (%struct.XXContextRec*, %struct.XXVArrayRec*)*, i32 (%struct.XXContextRec*, %struct.XXVArrayRec*, i32, i8*, i8 zeroext )*, i32 (%struct.XXContextRec*, %struct.XXVArrayRec*)*, void (%struct.XXContextRec*, %struct.XXVArrayRec*)*, i32 (%struct.XXContextRec*, %struct.XXFenceRec**)*, i32 (%struct.XXContextRec*, %struct.XXFenceRec*)*, i8 (%struct.XXContextRec*, i32, i8*) zeroext *, i32 (%struct.XXContextRec*, i32, i8*)*, i32 (%struct.XXContextRec*, %struct.XXQueryRec**)*, i32 (%struct.XXContextRec*, %struct.XXQueryRec*)*, i32 (%struct.XXContextRec*, %struct.XXQueryRec*, i32, i32*)*, i32 (%struct.XXContextRec*, %struct.XXBufferRec**, %struct.XXBufferData*, %struct.XXPluginBufferData*)*, i32 (%struct.XXContextRec*, %struct.XXBufferRec*)*, void (%struct.XXContextRec*, %struct.XXBufferRec*, i8*, i32)*, void (%struct.XXContextRec*, %struct.XXBufferRec*)*, void (%struct.XXContextRec*, %struct.XXBufferRec*)*, void (%struct.XXContextRec*, %struct.XXBufferRec*, i8**)*, void (%struct.XXContextRec*, %struct.XXBufferRec*, i8*)*, void (%struct.XXContextRec*, i8*)*, i8 (%struct.XXContextRec*, i8*) zeroext *, void (%struct.XXContextRec*, i8*)*, i32 (%struct.XXContextRec*, %struct.XXFramebufferRec**, %struct.XXFramebufferData*, %struct.XXPluginFramebufferData*)*, void (%struct.XXContextRec*, %struct.XXFramebufferRec*)*, i32 (%struct.XXContextRec*, %struct.XXFramebufferRec*)*, i32 (%struct.XXContextRec*, i32, i8*, i32, i8 zeroext )*, i32 (%struct.XXContextRec*, i32, i8*, i32, i8*)* }
	%struct.XXFallback = type { float*, %struct.XXRenderDispatch*, %struct.XXConfig*, i8*, i8*, i32, i32 }
	%struct.XXFenceRec = type opaque
	%struct.XXFixedFunctionProgram = type { %struct.PPStreamToken* }
	%struct.XXFogMode = type { %struct.GGC4, float, float, float, float, float, i16, i16, i16, i8, i8 }
	%struct.XXFormat = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, i32 }
	%struct.XXFragmentProgramLimits = type { i32, i32, i32, i16, i16, i32 }
	%struct.XXFramebufferAttachment = type { i32, i32, i32, i32 }
	%struct.XXFramebufferData = type { [10 x %struct.XXFramebufferAttachment], [8 x i16], i16, i16, i16, i8, i8, i32, i32 }
	%struct.XXFramebufferRec = type { %struct.XXFramebufferData*, %struct.XXPluginFramebufferData*, %struct.XXPixelFormat, i8, i8, i8, i8 }
	%struct.XXGeometryShaderLimits = type { i32, i32, i32, i32, i32, i32, i32 }
	%struct.XXHintMode = type { i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.XXHistogram = type { %struct.XXFramebufferAttachment*, i32, i16, i8, i8 }
	%struct.XXImagingCScale = type { %struct.GLTCoord2, %struct.GLTCoord2, %struct.GLTCoord2, %struct.GLTCoord2 }
	%struct.XXImagingSubset = type { %struct.XXConvolution, %struct.XXConvolution, %struct.XXConvolution, %struct.XXCMatrix, %struct.XXMinmax, %struct.XXHistogram, %struct.XXImagingCScale, %struct.XXImagingCScale, %struct.XXImagingCScale, %struct.XXImagingCScale, i32 }
	%struct.XXLight = type { %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.XXPointLineLimits, float, float, float, float, float, %struct.XXPointLineLimits, float, %struct.XXPointLineLimits, float, %struct.XXPointLineLimits, float, float, float, float, float }
	%struct.XXLightModel = type { %struct.GGC4, [8 x %struct.XXLight], [2 x %struct.XXMaterial], i32, i16, i16, i16, i8, i8, i8, i8, i8, i8 }
	%struct.XXLightProduct = type { %struct.GGC4, %struct.GGC4, %struct.GGC4 }
	%struct.XXLineMode = type { float, i32, i16, i16, i8, i8, i8, i8 }
	%struct.XXLogicOp = type { i16, i8, i8 }
	%struct.XXMaskMode = type { i32, [3 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXMaterial = type { %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.GGC4, float, float, float, float, [8 x %struct.XXLightProduct], %struct.GGC4, [6 x i32], [2 x i32] }
	%struct.XXMinmax = type { %struct.XXMinmaxTable*, i16, i8, i8 }
	%struct.XXMinmaxTable = type { %struct.GGC4, %struct.GGC4 }
	%struct.XXMipmaplevel = type { [4 x i32], [4 x i32], [4 x float], [4 x i32], i32, i32, float*, i8*, i16, i16, i16, i16, [2 x float] }
	%struct.XXMultisample = type { float, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXPipelineProgramData = type { i16, i8, i8, i32, %struct.PPStreamToken*, i64, %struct.GGC4*, i32 }
	%struct.XXPipelineProgramLimits = type { i32, i16, i16, i32, i16, i16, i32, i32 }
	%struct.XXPipelineProgramRec = type { %struct.XXPipelineProgramData*, %struct.PPStreamToken*, %struct.XXContextRec*, { %struct._LLFunction*, \2, \2, [20 x i32], [64 x i32], i32, i32, i32 }*, i32, i32 }
	%struct.XXPipelineProgramState = type { i8, i8, i8, i8, %struct.GGC4* }
	%struct.XXPixelFormat = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXPixelMap = type { i32*, float*, float*, float*, float*, float*, float*, float*, float*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.XXPixelMode = type { float, float, %struct.XXPixelStore, %struct.XXPixelTransfer, %struct.XXPixelMap, %struct.XXImagingSubset, i32, i32 }
	%struct.XXPixelPack = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.XXPixelStore = type { %struct.XXPixelPack, %struct.XXPixelPack }
	%struct.XXPixelTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.XXPluginBufferData = type { i32 }
	%struct.XXPluginFramebufferData = type { [10 x %struct.XXTextureRec*], i32, i32 }
	%struct.XXPluginProgramData = type { [3 x %struct.XXPipelineProgramRec*], %struct.XXBufferRec**, i32 }
	%struct.XXPluginState = type { [16 x [5 x %struct.XXTextureRec*]], [3 x %struct.XXTextureRec*], [3 x %struct.XXPipelineProgramRec*], %struct.XXProgramRec*, %struct.XXVArrayRec*, [16 x %struct.XXBufferRec*], %struct.XXFramebufferRec*, %struct.XXFramebufferRec* }
	%struct.XXPluginVArrayData = type { [32 x %struct.XXBufferRec*], %struct.XXBufferRec*, { i64 } }
	%struct.XXPointLineLimits = type { float, float, float }
	%struct.XXPointMode = type { float, float, float, float, %struct.XXPointLineLimits, float, i8, i8, i8, i8, i16, i16, i32, i16, i16 }
	%struct.XXPolygonMode = type { [128 x i8], float, float, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXProgramData = type { i32, i32, i32, i32, %struct.PPStreamToken*, i32*, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.XXProgramLimits = type { i32, i32, i32 }
	%struct.XXProgramRec = type { %struct.XXProgramData*, %struct.XXPluginProgramData*, %struct.GGC4**, i32 }
	%struct.XXQueryRec = type { i32, i32, %struct.XXQueryRec* }
	%struct.XXRect = type { i32, i32, i32, i32, i32, i32 }
	%struct.XXRegisterCombiners = type { i8, i8, i8, i8, i32, [2 x %struct.GGC4], [8 x %struct.XXRegisterCombinersPerStageState], %struct.XXRegisterCombinersFinalStageState }
	%struct.XXRegisterCombinersFinalStageState = type { i8, i8, i8, i8, [7 x %struct.XXRegisterCombinersPerVariableState] }
	%struct.XXRegisterCombinersPerPortionState = type { [4 x %struct.XXRegisterCombinersPerVariableState], i8, i8, i8, i8, i16, i16, i16, i16, i16, i16 }
	%struct.XXRegisterCombinersPerStageState = type { [2 x %struct.XXRegisterCombinersPerPortionState], [2 x %struct.GGC4] }
	%struct.XXRegisterCombinersPerVariableState = type { i16, i16, i16, i16 }
	%struct.XXRenderDispatch = type { void (%struct.XXContextRec*, i32, float)*, void (%struct.XXContextRec*, i32)*, i8 (%struct.XXContextRec*, i32, i32, i32, i32, i32, i32, i8*, i8 zeroext , %struct.XXBufferRec*) zeroext *, i8 (%struct.XXContextRec*, %struct.XXV*, i32, i32, i32, i32, i8*, i8 zeroext , %struct.XXBufferRec*) zeroext *, void (%struct.XXContextRec*, %struct.XXV*, i32, i32, i32, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32, float, float, i8*, i8 zeroext )*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV**, i32)*, void (%struct.XXContextRec*, %struct.XXV**, i32, i32)*, void (%struct.XXContextRec*, %struct.XXV**, i32, i32)*, void (%struct.XXContextRec*, i8*, i32, i32, i32, i32, i8*)*, i8* (%struct.XXContextRec*, i32, i32*)*, void (%struct.XXContextRec*, i32, i32, i32)*, i8* (%struct.XXContextRec*, i32, i32, i32, i32, i32)*, void (%struct.XXContextRec*, i32, i32, i32, i32, i32, i8*)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*, %struct.XXFenceRec*)*, void (%struct.XXContextRec*, i32, %struct.XXQueryRec*)*, void (%struct.XXContextRec*, %struct.XXQueryRec*)*, i8 (%struct.XXContextRec*, i32, i32, i32, i32, i32, i8*, %struct.GGC4*, %struct.XXCurrent16A*) zeroext *, i8 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i32, i32, i32, i32, i32, i32, i32) zeroext *, i8 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8 zeroext , %struct.XXBufferRec*) zeroext *, i8 (%struct.XXContextRec*, %struct.XXTextureRec*, i32) zeroext *, i8 (%struct.XXContextRec*, %struct.XXBufferRec*, i32, i32, i8*) zeroext *, void (%struct.XXContextRec*, i32)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* }
	%struct.XXRenderFeatures = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXScissorTest = type { %struct.XXFramebufferAttachment, i8, i8, i8, i8 }
	%struct.XXSharedRec = type { %struct.pthread_mutex_t, i32 }
	%struct.XXState = type { i16, i16, i16, i16, i32, i32, [256 x %struct.GGC4], [128 x %struct.GGC4], %struct.XXViewport, %struct.XXXf, %struct.XXLightModel, %struct.XXTextureTargets, %struct.XXAlphaTest, %struct.XXBlendMode, %struct.XXClearC, %struct.XXCBuffer, %struct.XXDepthTest, %struct.XXArrayRange, %struct.XXFogMode, %struct.XXHintMode, %struct.XXLineMode, %struct.XXLogicOp, %struct.XXMaskMode, %struct.XXPixelMode, %struct.XXPointMode, %struct.XXPolygonMode, %struct.XXScissorTest, i32, %struct.XXStencilTest, [8 x %struct.XXTextureMode], [8 x %struct.XXTextureMode], [16 x %struct.XXTextureImageMode], %struct.XXArrayRange, [8 x %struct.XXTextureCoordGen], %struct.XXClipPlane, %struct.XXMultisample, %struct.XXRegisterCombiners, %struct.XXArrayRange, %struct.XXArrayRange, [3 x %struct.XXPipelineProgramState], %struct.XXXfFeedback, i32*, %struct.XXFixedFunctionProgram, [3 x i32] }
	%struct.XXStencilTest = type { [3 x { i32, i32, i16, i16, i16, i16 }], i32, [4 x i8] }
	%struct.XXStippleData = type { i32, i16, i16, [32 x [32 x i8]] }
	%struct.XXTextureCoordGen = type { { i16, i16, %struct.GGC4, %struct.GGC4 }, { i16, i16, %struct.GGC4, %struct.GGC4 }, { i16, i16, %struct.GGC4, %struct.GGC4 }, { i16, i16, %struct.GGC4, %struct.GGC4 }, i8, i8, i8, i8 }
	%struct.XXTextureGeomState = type { i16, i16, i16, i16, i16, i8, i8, i8, i8, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [6 x i16], [6 x i16] }
	%struct.XXTextureImageMode = type { float }
	%struct.XXTextureLevel = type { i32, i32, i16, i16, i16, i8, i8, i16, i16, i16, i16, i8* }
	%struct.XXTextureLimits = type { float, float, i16, i16, i16, i16, i16, i16, i16, i16, i16, i8, i8, [8 x i32], i32 }
	%struct.XXTextureMode = type { %struct.GGC4, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, float, float, i16, i16, i16, i16, i16, i16, [4 x i16], i8, i8, i8, i8, [3 x float], [4 x float], float, float }
	%struct.XXTextureParamState = type { i16, i16, i16, i16, i16, i16, %struct.GGC4, float, float, float, float, i16, i16, i16, i16, float, i16, i8, i8, i32, i8* }
	%struct.XXTextureRec = type { [4 x float], %struct.XXTextureState*, %struct.XXMipmaplevel*, %struct.XXMipmaplevel*, float, float, float, float, i32, i32, i32, i32, i32, i32, i32, [2 x %struct.PPStreamToken] }
	%struct.XXTextureState = type { i16, i8, i8, i16, i16, float, i32, %struct.GGSWRSurface*, %struct.XXTextureParamState, %struct.XXTextureGeomState, %struct.XXTextureLevel, [6 x [15 x %struct.XXTextureLevel]] }
	%struct.XXTextureTargets = type { i64, i64, i64, i64, i64, i64 }
	%struct.XXXf = type { [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, float, i8, i8, i8, i8, i32, i32, i32, i16, i16, i8, i8, i8, i8, i32 }
	%struct.XXXfFeedback = type { i8, i8, i8, i8, [16 x i32], [16 x i32] }
	%struct.XXXfFeedbackLimits = type { i32, i32, i32, i32, i32 }
	%struct.XXV = type { %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.XXPointLineLimits, float, %struct.GGC4, float, float, float, i8, i8, i8, i8, i32, i32, i32, i32, [4 x float], [2 x %struct.XXMaterial*], i32, i32, [8 x %struct.GGC4] }
	%struct.XXVArrayData = type { [32 x %struct.XXVArrayElement], { i64 }, { i64 }, i16, i16, i32, i8*, i8, i8, i8, i8 }
	%struct.XXVArrayElement = type { i8*, i8*, i32, i16, i8, i8 }
	%struct.XXVArrayRec = type opaque
	%struct.XXVArrayTypes = type { i16, i16, i16, i16, i16, i16 }
	%struct.XXVDescriptor = type { i8, i8, i8, i8, [0 x i32] }
	%struct.XXVProgramLimits = type { i16, i16 }
	%struct.XXViewport = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, double, double, i32, i32, i32, i32, float, float, float, float }
	%struct.XXViewportConfig = type { %struct.GLGXfKey, %struct.GLGXfKey, %struct.GLGXfKey }
	%struct.GLEEnableHashObject = type { i32, void (%struct.__GGContextRec*, i32, i32)*, %struct.GLEEnableHashObject*, i8* }
	%struct.GLEPContextData = type { %struct.XXContextRec*, %struct.XXConfig, %struct.XXPluginState, %struct.GLEPPlugin*, i32 }
	%struct.GLEPPlugin = type { %struct.GLEPPlugin*, [256 x i8], i8*, i32, i32, %struct.XXDispatch }
	%struct.GLGCTable = type { i32, i32, i32, i8* }
	%struct.GLGOperation = type { i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, %struct.GLGCTable, %struct.GLGCTable, %struct.GLGCTable }
	%struct.GLGProcessor = type { void (%struct.XXPixelMode*, %struct.GLGOperation*, %struct._GLGFunctionKey*)*, %struct._LLFunction*, %struct._GLGFunctionKey* }
	%struct.GLGXfKey = type { i32, i32 }
	%struct.GGAttrib = type { %struct.GGAttribPixelMode }
	%struct.GGAttribPixelMode = type { float, float, i32, %struct.GGPixelTransfer, %struct.GGPixelMap }
	%struct.GGClientAttrib = type { %struct.GGClientAttribVArray }
	%struct.GGClientAttribVArray = type { [32 x %struct.GLGCTable], i16, i16, i32, i8*, i32, i32, i32, [32 x i32], i32, i32, i32, i8*, i32, i32, i32, i32 }
	%struct.GGC4 = type { float, float, float, float }
	%struct.GGPixelFormat = type { %struct.GGPixelFormat*, i32, i32, i32, i32, i32, i32, i32, i16, i16, i16, i16, i32, i8, i8, i8, i8, i32 }
	%struct.GGPixelMap = type { [256 x i32], [256 x float], [256 x float], [256 x float], [256 x float], [256 x float], [256 x float], [256 x float], [256 x float], [256 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.GGPixelTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32 }
	%struct.GGRendererInfo = type { %struct.GGRendererInfo*, i32, i32, i32, i32, i32, i32, i32, i16, i16, i16, i16, i16, i8, i8, i32, i32, i32 }
	%struct.GGSWRSurface = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i8*, [4 x i8*], i32 }
	%struct.GLSBuffer = type { i8* }
	%struct.GLSDrawable = type { %struct.GLSWindowRec* }
	%struct.GLSWindowRec = type { %struct.GLGXfKey, %struct.GLGXfKey, i32, i32, %struct.GLSDrawable, [2 x i8*], i8*, i8*, i8*, [4 x i8*], i32, i32, i32, i32, [4 x i32], i16, i16, i16, i8, i8, %struct.XXProgramLimits, i32, i32, i8*, i8* }
	%struct.GLTCoord2 = type { float, float }
	%struct.LLFPContext = type { float, i32, i32, i32, float, [3 x float] }
	%struct.LLFragmentAttrib = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, [8 x <4 x float>] }
	%struct.LLProgramData = type { %struct._LLFunction*, %struct.LLProgramData*, %struct.LLProgramData*, [64 x i32], i32, i32 }
	%struct.LLSubmitKey = type { %struct.GLGXfKey }
	%struct.LLTextures = type { [16 x %struct.XXTextureRec*] }
	%struct.LFSStream = type { %struct.LFSStreamChunkList*, { %struct.LFSStreamChunkList*, %struct.LFSStreamChunkList* }, { %struct.LFSStreamChunkList*, %struct.LFSStreamChunkList* }, %struct.LFSStreamChunkList* }
	%struct.LFSStreamChunk = type { i8, i8, i8, i8, { %struct.LFSStreamOperation }, i8*, i8* }
	%struct.LFSStreamChunkList = type { %struct.LFSStreamChunk*, %struct.LFSStreamChunk*, i32 }
	%struct.LFSStreamOperation = type { i32, %struct.XXFramebufferAttachment, [3 x %struct.XXFramebufferAttachment] }
	%struct.PPStreamToken = type { { i16, i16, i32 } }
	%struct._GLGFunctionKey = type { %struct.LLSubmitKey }
	%struct._GLPHashMachine = type opaque
	%struct._GLPHashObject = type opaque
	%struct._LLConstants = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, float, float, float, float, float, float, float, float, float, float, float, float, [256 x float], [528 x i8] }
	%struct._LLFunction = type opaque
	%struct.__GGContextRec = type { %struct.__GLconstants16A, %struct.__GLviewport16A, %struct.XXCurrent16A, %struct.__GLlightmodel16A, [8 x i32], [32 x i32], [32 x i32], [32 x i32], [8 x i32], [8 x i32], %struct.XXState, %struct.__GLglobals16A*, [3 x %struct.__GLindexoffsets*], [16 x float]*, %struct.__GLmatrixidentifiers, %struct.GLEEnableHashObject*, %struct.XXPluginBufferData, %struct.XXPluginBufferData, %struct.LLTextures, %struct.GLGProcessor, %struct.__GLVmachine, %struct.__GLVarray, %struct.__GLVarraymachine, %struct.__GLinterpolate, %struct.__GLprimitive, %struct.__GLviewport, i32, %struct.__GLhashobject*, %struct.__GLhashobject*, %struct.__GLhashobject*, %struct.__GLtransform_feedback, %struct.__GLhashobject*, %struct.__GLclipplane, %struct.__GLselect, %struct.__GLfeedback, %struct.XXArrayRange, %struct.XXLogicOp, %struct.__GLmatrixmachine, %struct.__GLattribmachine, %struct.__GLshared*, %struct.__GLtexturemachine, %struct.__GLlistmachine, %struct.__GLpipelineprogrammachine, %struct.__GLshadermachine, %struct.__GLmachineshared, %struct.__GLquerymachine, %struct.__GLcurrentindex, %struct.__GLcmdbufmachine, %struct.__GLclientState, %struct.__GLeval, %struct.__GLmapdata*, %struct.__GLcoefficients*, %struct.__GLdrawpixelsobject, %struct.__GLorphanlist, %struct.__GLorphanlist*, %struct.__GLhashobject*, %struct.__GLhashobject*, %struct.__GLhashobject*, i8*, i8*, %struct.XXRenderDispatch*, i8*, i8*, i16, i8, i8, i32, i32, i32, i32, i32, i32, i32, [1 x i32], %struct.XXContextRec*, %struct.XXConfig*, %struct.XXRenderDispatch, %struct.XXViewportConfig, %struct.XXMinmaxTable, %struct.XXDispatch, i8*, i32, i32, %struct.__GGContextRec*, [4096 x i8], i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [2 x i32], [0 x %struct.GLEPContextData] }
	%struct.__GLarrayelementDrawInfoListType = type { i32, [32 x i8] }
	%struct.__GLattriblink = type { i32, %struct.GGAttrib, %struct.__GLattriblink* }
	%struct.__GLattribmachine = type { [16 x %struct.__GLattriblink*], %struct.__GLattriblink**, [16 x %struct.__GLclientattriblink*], %struct.__GLclientattriblink**, i32, i32 }
	%struct.__GLbufferobjectclient = type { %struct.XXBufferData, i8, i8, i8, i8, i32, i8, i8, i8, i8, %struct.__GLhashobject* }
	%struct.__GLclientState = type { %struct.__GLVarrayobjectclient*, %struct.__GLVarrayobjectclient*, %struct.__GLbufferobjectclient*, %struct.__GLbufferobjectclient*, %struct.__GLbufferobjectclient*, %struct.__GLbufferobjectclient*, i16, i16, i32, %struct.XXPixelMode, %struct.__GLmachineshared, %struct.__GLmachineshared, %struct.__GLpagehashuint32* }
	%struct.__GLclientattriblink = type { i32, %struct.GGClientAttrib, %struct.__GLclientattriblink* }
	%struct.__GLclipplane = type { i32, [6 x i32], i8, i8, i8, i8 }
	%struct.__GLcmdbufmachine = type { i32, i32, %struct.__GLcommandbuf*, %struct.__GLcommandbuf*, %struct.__GLcommandbuf* (%struct.__GGContextRec*, i32)*, [32 x %struct.__GLcommandbuf*], i32, i8*, i8*, i8*, i8*, %struct._opaque_pthread_t*, %struct.pthread_mutex_t, %struct.pthread_mutex_t, %struct.pthread_mutex_t, %struct.pthread_mutex_t, %struct.pthread_cond_t, %struct.pthread_cond_t, %struct.pthread_cond_t, %struct.pthread_mutex_t, %struct.pthread_cond_t, i32, i32, i32, i8, i8, i8, i8 }
	%struct.__GLcoefficients = type { float, float, i32, i32, i32, i32, [10 x float], [10 x float], [10 x float], [10 x float] }
	%struct.__GLcommandbuf = type { %struct.__GLcommandelem*, %struct.__GLcommandelem*, i32, [0 x i8] }
	%struct.__GLcommandelem = type { i32 (%struct.__GGContextRec*, i8*)*, i32, [0 x i8] }
	%struct.__GLconstants16A = type { [4 x i32], [4 x i32], [4 x float], [4 x i32], [4 x i32], [4 x i32], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], [4 x float], float, float, float, float, float, float, float, float, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i8*, i8*, i8*, %struct._LLConstants* }
	%struct.__GLcurrentindex = type { float, [2 x %struct.XXPointLineLimits] }
	%struct.__GLdrawpixelsobject = type { i32, i32, i32, i32, i32, i32, %struct.__GLhashobject**, i8* }
	%struct.__GLeval = type { %struct.__GLmapgrid1, %struct.__GLmapgrid2, i32, i32, %struct.__GLarrayelementDrawInfoListType, %struct.__GLarrayelementDrawInfoListType, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.__GLfeedback = type { float*, float*, i32, i32, i8, i8, i8, i8 }
	%struct.__GLfragmentprogram = type { %struct.LFSStream*, i32, i16, i16, i16, i16 }
	%struct.__GLglobals16A = type { [256 x float], [48 x i8] }
	%struct.__GLhashallocation = type { %struct.__GLhashallocation*, i32, i32 }
	%struct.__GLhashcommon = type { %struct.__GLhashobject*, i8*, void (%struct.__GGContextRec*, %struct.__GLhashobject*)*, i32, i32, i32 }
	%struct.__GLhashobject = type { %struct.__GLhashcommon, { %struct.__GLprogram } }
	%struct.__GLindexoffsets = type { [1024 x i16], [256 x i16], [8 x %struct.XXHintMode], [8 x [2 x { i16, i16, i16 }]], [2 x %struct.XXVArrayTypes], [8 x [4 x i16]], [8 x [4 x i16]], [8 x i16], [6 x i16], i16, i16, i16, i16, i16, i16, i16, [24 x [4 x i16]], [24 x [4 x i16]], [24 x [4 x i16]], [24 x [4 x i16]], i16 }
	%struct.__GLinterpolate = type { void (%struct.__GGContextRec*, %struct.__GLV*, %struct.__GLV*, %struct.__GLV*, float)*, i8 (%struct.__GGContextRec*, %struct.__GLV*, %struct.__GLV*, %struct.__GLV*, i8 zeroext ) zeroext *, %struct._LLFunction*, %struct._LLFunction*, i32, i32 }
	%struct.__GLlight = type { float, float, float, float, float, float, i8, i8, i8, i8, i8, i8, i8, i8, [11 x float], float }
	%struct.__GLlightmodel16A = type { [8 x %struct.__GLlight], [2 x %struct.__GLmaterial*], %struct.__GLmaterial*, i32, i8, i8, i8, i8, i32, void (%struct.__GGContextRec*, %struct.GGC4*)*, void (%struct.__GGContextRec*, %struct.__GLV*)*, [2 x void (%struct.__GGContextRec*, %struct.__GLV*, i32)*], [2 x void (%struct.__GGContextRec*, %struct.__GLV**, i32)*] }
	%struct.__GLlistmachine = type { %struct.__GLhashobject*, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, %struct.__GLhashobject* }
	%struct.__GLmachineshared = type { %struct.__GLhashobject**, %struct.__GLhashallocation*, i32, i32 }
	%struct.__GLmap1color = type { i32, float, float, float, [10 x %struct.GGC4] }
	%struct.__GLmap1index = type { i32, float, float, float, [10 x float] }
	%struct.__GLmap1normal = type { i32, float, float, float, [10 x %struct.XXPointLineLimits] }
	%struct.__GLmap1texture1 = type { i32, float, float, float, [10 x %struct.XXTextureImageMode] }
	%struct.__GLmap1texture2 = type { i32, float, float, float, [10 x %struct.GLTCoord2] }
	%struct.__GLmap1VattribN = type { %struct.__GLmap1color }
	%struct.__GLmap2color = type { i32, i32, float, float, float, float, float, float, [100 x %struct.GGC4] }
	%struct.__GLmap2index = type { i32, i32, float, float, float, float, float, float, [100 x float] }
	%struct.__GLmap2normal = type { i32, i32, float, float, float, float, float, float, [100 x %struct.XXPointLineLimits] }
	%struct.__GLmap2texture1 = type { i32, i32, float, float, float, float, float, float, [100 x %struct.XXTextureImageMode] }
	%struct.__GLmap2texture2 = type { i32, i32, float, float, float, float, float, float, [100 x %struct.GLTCoord2] }
	%struct.__GLmap2VattribN = type { %struct.__GLmap2color }
	%struct.__GLmapdata = type { %struct.__GLmap1normal, %struct.__GLmap1color, %struct.__GLmap1normal, %struct.__GLmap1color, %struct.__GLmap1index, %struct.__GLmap1texture1, %struct.__GLmap1texture2, %struct.__GLmap1normal, %struct.__GLmap1color, %struct.__GLmap2normal, %struct.__GLmap2color, %struct.__GLmap2normal, %struct.__GLmap2color, %struct.__GLmap2index, %struct.__GLmap2texture1, %struct.__GLmap2texture2, %struct.__GLmap2normal, %struct.__GLmap2color, [16 x %struct.__GLmap1VattribN], [16 x %struct.__GLmap2VattribN], [16 x i8], [16 x i8] }
	%struct.__GLmapgrid1 = type { i32, float, float, float }
	%struct.__GLmapgrid2 = type { i32, float, float, float, i32, float, float, float }
	%struct.__GLmaterial = type { %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.GGC4, float, float, float, float, [8 x %struct.XXLightProduct], %struct.GGC4, i32, i32, %struct.__GLsum, %struct.__GLmaterial*, %struct.__GLmaterial* }
	%struct.__GLmatrixidentifiers = type { i32, i32, i32, i32, [8 x i32] }
	%struct.__GLmatrixmachine = type { [16 x float]*, [16 x float]*, i32*, [24 x i32], i16 (%struct.__GGContextRec*, %struct.__GLV*, i32) zeroext *, void (%struct.__GGContextRec*, %struct.__GLV*, i32)*, void (%struct.__GGContextRec*, %struct.__GLV*, i32)*, void (%struct.__GGContextRec*, %struct.__GLV*, i32)*, i16 (%struct.__GGContextRec*, %struct.__GLV*, i32) zeroext *, void (%struct.__GGContextRec*, %struct.__GLV*, i32)*, void (%struct.__GGContextRec*, %struct.__GLV*, i32)*, void (%struct.__GGContextRec*, %struct.__GLV*, i32)*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, i32, i32, [3 x i32], [3 x i32], i32, i32, [3 x i32], [3 x i32], i32, i32, i32, i8, i8, i8, i8 }
	%struct.__GLmemoryobject = type { i8*, i32, i16, i8, i8, i32, [0 x i8*] }
	%struct.__GLorphanlist = type { i32, i32, i32, i32, %struct.__GLorphannode*, %struct.__GLorphannode }
	%struct.__GLorphannode = type { %struct.__GLorphannode*, %struct.__GLorphannode*, %struct.__GLorphannode*, %struct.__GLorphannode*, %struct.__GLmemoryobject }
	%struct.__GLpagehashlistuint32 = type { %struct.__GLpagehashnodeuint32*, %struct.__GLpagehashnodeuint32* }
	%struct.__GLpagehashnodeuint32 = type { %struct.__GLpagehashnodeuint32*, %struct.__GLpagehashnodeuint32*, i32, [5 x i32], [0 x i8] }
	%struct.__GLpagehashuint32 = type { i32, i32, i32, %struct.__GLpagehashlistuint32** }
	%struct.__GLpipelineprogram = type { %struct.__GLpipelineprogramcommon, { %struct.__GLfragmentprogram } }
	%struct.__GLpipelineprogramcommon = type { [8 x %struct.XXPipelineProgramRec*], %struct.LLProgramData*, i8*, i32, i32, i32, [32 x i32], i8, i8, i16, i16, i16, i16, i16, i16, i16, %struct.__GLindexoffsets, %struct.XXPipelineProgramData }
	%struct.__GLpipelineprogrammachine = type { i8*, i16 (%struct.__GGContextRec*, %struct.__GLV*, i32) zeroext *, %struct._LLFunction*, %struct.PPStreamToken*, %struct.GGC4*, i16 (%struct.__GGContextRec*, %struct.__GLV*, i32) zeroext *, %struct._LLFunction*, %struct.PPStreamToken*, %struct.GGC4*, [3 x %struct.__GLhashobject*], [3 x %struct.__GLhashobject*], [3 x %struct.__GLpipelineprogram*], [3 x %struct._GLPHashMachine*], [3 x %struct._GLPHashObject*], i32, i8, i8, i8, i8, i16, i16, [4 x i8] }
	%struct.__GLprimitive = type { [14 x void (%struct.__GGContextRec*)*], void (%struct.__GGContextRec*)*, void (%struct.__GGContextRec*)*, void (%struct.__GGContextRec*)*, i16, i16, i32, i8, i8, i8, i8, i32 }
	%struct.__GLprogram = type { [8 x %struct.XXProgramRec*], i8*, %struct.GGC4*, i8*, %struct.__GLhashobject**, [3 x %struct.__GLshadercompilelink], i32, i32, i32, i32, i32, i32, %struct.__GLhashobject**, i32*, %struct.GGC4**, i32, i32, %struct.XXProgramData, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, %struct.PPStreamToken*, [0 x %struct.XXPluginProgramData] }
	%struct.__GLquerymachine = type { %struct.__GLmachineshared, %struct.__GLhashobject*, %struct.__GLhashobject*, %struct.__GLhashobject*, i32, i32, i32 }
	%struct.__GLselect = type { i32*, i32*, [128 x i32], i32*, i32*, i32, i32, i8, i8, i8, i8 }
	%struct.__GLshadercompilelink = type { i32, i8, i8, i8, i8, i32, [32 x i32], [16 x i32], [16 x i16], %struct.PPStreamToken*, %struct.LLProgramData*, i32*, %struct.XXPipelineProgramData, %struct.__GLindexoffsets }
	%struct.__GLshadermachine = type { %struct.__GLhashobject*, float**, float**, i32 }
	%struct.__GLshared = type { [8 x %struct.__GLmachineshared], i32, i32, i8, i8, i8, i8, %struct.pthread_mutex_t, [8 x %struct.XXSharedRec*] }
	%struct.__GLsum = type { %struct.XXPointLineLimits, i8, i8, i8, i8 }
	%struct.__GLtexturemachine = type { [16 x [5 x %struct.__GLhashobject*]], [5 x %struct.__GLhashobject*], [5 x %struct.__GLhashobject*], [3 x %struct.__GLhashobject*], [3 x %struct.__GLhashobject*], i64, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.__GLtransform_feedback = type { [16 x %struct.__GLhashobject*], i16, i16 }
	%struct.__GLV = type { %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.GGC4, %struct.XXPointLineLimits, float, %struct.GGC4, float, float, float, i8, i8, i8, i8, i32, i32, i32, i32, [4 x float], [2 x %struct.__GLmaterial*], i32, i32, [8 x %struct.GGC4] }
	%struct.__GLVarray = type { i8*, i8*, %struct.__GLarrayelementDrawInfoListType*, i8*, %struct.__GLhashobject*, [32 x i8*], i8*, i32, i32, i32, i32, i32, i16, i16, i32, i32 }
	%struct.__GLVarraymachine = type { %struct.__GLhashobject*, %struct.__GLhashobject* }
	%struct.__GLVarrayobjectclient = type { %struct.__GLbufferobjectclient*, %struct.__GLbufferobjectclient*, [32 x %struct.__GLbufferobjectclient*], i64, i64, %struct.XXVArrayData, %struct.__GLhashobject* }
	%struct.__GLVmachine = type { %struct.__GLV*, %struct.__GLV*, %struct.__GLV*, %struct.__GLV*, %struct.__GLV*, i8*, i8*, %struct.__GLV*, %struct.__GLV**, %struct.__GLV**, %struct.__GLV**, %struct.__GLV**, %struct.__GLV**, i32, i32, i32, i32, i32, i16, i16, i16, i16, void (%struct.__GGContextRec*, float, float, float, float, %struct.__GLV*)*, void (%struct.__GGContextRec*, float, float, float, float, %struct.__GLV*)*, void (%struct.__GGContextRec*, i32)*, void (%struct.__GGContextRec*, i32)*, void (%struct.__GGContextRec*, i8*, i8*, i32)*, void (%struct.__GGContextRec*, i8*, i32)*, void (%struct.__GGContextRec*, i32)*, void (%struct.__GGContextRec*, i32)*, void (%struct.__GGContextRec*, i32)*, void (%struct.__GGContextRec*, i32)*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, %struct._LLFunction*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32*, [32 x i8], [32 x i8], %struct.XXVDescriptor*, %struct.XXVDescriptor* }
	%struct.__GLviewport = type { float, float, float, float, float }
	%struct.__GLviewport16A = type { float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.__darwin_pthread_handler_rec = type { void (i8*)*, i8*, %struct.__darwin_pthread_handler_rec* }
	%struct._opaque_pthread_t = type { i32, %struct.__darwin_pthread_handler_rec*, [596 x i8] }
	%struct.pthread_cond_t = type { i32, [24 x i8] }
	%struct.pthread_mutex_t = type { i32, [40 x i8] }

define i16 @f(%struct.__GGContextRec* %ctx, %struct.__GLV* %stack, i32 %num_vtx) {
entry:
	alloca [4 x <4 x float>]		; <[4 x <4 x float>]*>:0 [#uses=167]
	alloca [4 x <4 x float>]		; <[4 x <4 x float>]*>:1 [#uses=170]
	alloca [4 x <4 x i32>]		; <[4 x <4 x i32>]*>:2 [#uses=12]
	%tmp3 = getelementptr %struct.__GGContextRec* %ctx, i32 0, i32 42, i32 3		; <%struct.PPStreamToken**> [#uses=1]
	%tmp4 = load %struct.PPStreamToken** %tmp3		; <%struct.PPStreamToken*> [#uses=13]
	%.sub6235.i = getelementptr [4 x <4 x float>]* %0, i32 0, i32 0		; <<4 x float>*> [#uses=76]
	%.sub.i = getelementptr [4 x <4 x float>]* %1, i32 0, i32 0		; <<4 x float>*> [#uses=59]

	%tmp116117.i1061.i = bitcast %struct.PPStreamToken* %tmp4 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%tmp124.i1062.i = getelementptr <4 x float>* %tmp116117.i1061.i, i32 63		; <<4 x float>*> [#uses=1]
	%tmp125.i1063.i = load <4 x float>* %tmp124.i1062.i		; <<4 x float>> [#uses=5]
	%tmp828.i1077.i = shufflevector <4 x float> %tmp125.i1063.i, <4 x float> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x float>> [#uses=4]
	%tmp704.i1085.i = load <4 x float>* %.sub6235.i		; <<4 x float>> [#uses=1]
	%tmp712.i1086.i = call <4 x float> @llvm.x86.sse.max.ps( <4 x float> %tmp704.i1085.i, <4 x float> %tmp828.i1077.i )		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp712.i1086.i, <4 x float>* %.sub.i

	%tmp2587.i1145.gep.i = getelementptr [4 x <4 x float>]* %1, i32 0, i32 0, i32 2		; <float*> [#uses=1]
	%tmp5334.i = load float* %tmp2587.i1145.gep.i		; <float> [#uses=5]
	%tmp2723.i1170.i = insertelement <4 x float> undef, float %tmp5334.i, i32 2		; <<4 x float>> [#uses=5]
	store <4 x float> %tmp2723.i1170.i, <4 x float>* %.sub6235.i

	%tmp1406.i1367.i = shufflevector <4 x float> %tmp2723.i1170.i, <4 x float> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x float>> [#uses=1]
	%tmp84.i1413.i = load <4 x float>* %.sub6235.i		; <<4 x float>> [#uses=1]
	%tmp89.i1415.i = mul <4 x float> %tmp84.i1413.i, %tmp1406.i1367.i		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp89.i1415.i, <4 x float>* %.sub.i
        ret i16 0
}

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>)
