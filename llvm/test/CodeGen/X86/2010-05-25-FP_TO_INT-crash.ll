; RUN: llc -O0 -march=x86 -mattr=+sse3 %s
; Formerly crashed - PR 7191 / 8023512
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0"

%0 = type { i16, i16, i32 }
%1 = type { i32, i32, i16, i16, i16, i16 }
%2 = type { i16, i16, %struct.GLTColor4, %struct.GLTColor4 }
%3 = type { void (i8*, i8*, i32, i8*)*, i32 (i8*, ...)*, i8* (%struct.GLDContextRec*, %struct.GLDFramebufferRec*, i8, i32, i32)* }
%struct.GLDActiveTextureTargets = type { i64, i64, i64, i64, i64, i64 }
%struct.GLDAlphaTest = type { float, i16, i8, i8 }
%struct.GLDArrayRange = type { i8, i8, i8, i8 }
%struct.GLDBlendMode = type { i16, i16, i16, i16, %struct.GLTColor4, i16, i16, i8, i8, i8, i8 }
%struct.GLDBufferData = type { i8*, i32, i32, i16, i16, i8, i8, i8, i8 }
%struct.GLDBufferRec = type { %struct.GLDBufferData*, %struct.GLDPluginBufferData* }
%struct.GLDBufferstate = type { %struct.GLTDimensions, %struct.GLTDimensions, %struct.GLTFixedColor4, %struct.GLTFixedColor4, i8, i8, i8, i8, [0 x i32], %union.GLSBuffer, %union.GLSBuffer, %union.GLSBuffer, [8 x %union.GLSBuffer], %union.GLSBuffer }
%struct.GLDClearColor = type { double, %struct.GLTColor4, %struct.GLTColor4, float, i32 }
%struct.GLDClipPlane = type { i32, [6 x %struct.GLTColor4] }
%struct.GLDColorBuffer = type { i16, i8, i8, [8 x i16], i8, i8, i8, i8 }
%struct.GLDColorMatrix = type { [16 x float]*, %struct.GLDImagingColorScale }
%struct.GLDConfig = type { i32, float, %struct.GLTDimensions, %struct.GLTDimensions, i8, i8, i8, i8, i8, i8, i16, i32, i32, i32, %struct.GLDPixelFormatInfo, %struct.GLDPointLineLimits, %struct.GLDPointLineLimits, %struct.GLDRenderFeatures, i32, i32, i32, i32, i32, i32, i32, i32, %struct.GLDMultisamplePositions, %struct.GLDTextureLimits, [3 x %struct.GLDPipelineProgramLimits], %struct.GLDFragmentProgramLimits, %struct.GLDVertexProgramLimits, %struct.GLDGeometryShaderLimits, %struct.GLDGeometryShaderLimits, %struct.GLDTransformFeedbackLimits, i16, i8, i8, %struct.GLDVertexDescriptor*, %struct.GLDVertexDescriptor*, [4 x i32], [8 x i32], %struct.GLDMultisamplePositions* }
%struct.GLDContextRec = type { float, float, float, float, float, float, float, float, %struct.GLTColor4, %struct.GLTColor4, %struct.GLVMFPContext, [16 x [2 x %union.PPStreamToken]], %struct.GLGProcessor, %struct._GLVMConstants*, void (%struct.GLDContextRec*, i32, i32, %struct.GLVMFragmentAttribRec*, %struct.GLVMFragmentAttribRec*, i32)*, %struct._GLVMFunction*, %union.PPStreamToken*, void (%struct.GLDContextRec*, %struct.GLDVertex*)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, %struct.GLDVertex*)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, %struct.GLDVertex*, %struct.GLDVertex*)*, %struct._GLVMFunction*, %struct._GLVMFunction*, %struct._GLVMFunction*, [4 x i32], [3 x i32], [3 x i32], %union.PPStreamToken, %struct.GLDConfig*, %struct.GLDFramebufferRec*, %struct.GLDFramebufferRec*, %struct.GLDBufferstate, %struct.GLDReadBufferstate, %struct.GLDLayeredBufferstate, [64 x %struct.GLTColor4*], %struct.GLDSharedRec*, %struct.GLDState*, %struct.GLDPluginState*, %struct.GLDVertex*, %struct.GLVMFragmentAttribRec*, %struct.GLVMFragmentAttribRec*, %struct.GLVMFragmentAttribRec*, %struct.GLDProgramRec*, %struct.GLDPipelineProgramRec*, %struct.GLVMTextures, %struct.GLDQueryRec*, %struct.GLDQueryRec*, %struct.GLDQueryRec*, %struct.GLTDimensions, i64 ()*, %struct.GLDFallback, %3, %union.GLSDrawable, i32, float, float, %struct.GLDRect, %struct.GLDFormat, %struct.GLDFormat, %struct.GLDFormat, %struct.GLDStippleData, i32, i32, i32, i32, i16, i8, i8, i8, i8, [2 x i8], [0 x i32] }
%struct.GLDConvolution = type { %struct.GLTColor4, %struct.GLDImagingColorScale, i16, i16, [0 x i32], float*, i32, i32 }
%struct.GLDCurrent = type { [8 x %struct.GLTColor4], [16 x %struct.GLTColor4], %struct.GLTColor4, %struct.GLDPointLineLimits, float, %struct.GLDPointLineLimits, float, [4 x float], float, float, float, i8, i8, i8, i8, i32, i32, i32, i32 }
%struct.GLDDepthTest = type { i16, i16, i8, i8, i8, i8, double, double }
%struct.GLDDitherMode = type { i8, i8, i8, i8 }
%struct.GLDDrawableOffscreen = type { i32, i32, i32, [0 x i32], i8* }
%struct.GLDDrawableWindow = type { i32, i32, i32 }
%struct.GLDFallback = type { float*, %struct.GLDRenderDispatch*, %struct.GLDConfig*, i8*, i8*, i32, i32 }
%struct.GLDFixedFunction = type { %union.PPStreamToken* }
%struct.GLDFogMode = type { %struct.GLTColor4, float, float, float, float, float, i16, i16, i16, i8, i8 }
%struct.GLDFormat = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, i32 }
%struct.GLDFragmentProgramLimits = type { i32, i32, i32, i16, i16, i32, i16, i16, i16, i16 }
%struct.GLDFramebufferAttachment = type { i16, i16, i32, i32, i32 }
%struct.GLDFramebufferData = type { [10 x %struct.GLDFramebufferAttachment], [8 x i16], i16, i16, i16, i8, i8, i32, i32, i32 }
%struct.GLDFramebufferRec = type { %struct.GLDFramebufferData*, %struct.GLDPluginFramebufferData*, [10 x %struct.GLDFormat], i8, i8, i16, [0 x i32] }
%struct.GLDGeometryShaderLimits = type { i32, i32, i32, i32, i32, i16, i16 }
%struct.GLDHintMode = type { i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
%struct.GLDHistogram = type { %struct.GLTFixedColor4*, i32, i16, i8, i8 }
%struct.GLDImagingColorScale = type { %struct.GLDMultisamplePositions, %struct.GLDMultisamplePositions, %struct.GLDMultisamplePositions, %struct.GLDMultisamplePositions }
%struct.GLDImagingSubset = type { %struct.GLDConvolution, %struct.GLDConvolution, %struct.GLDConvolution, %struct.GLDColorMatrix, %struct.GLDMinmax, %struct.GLDHistogram, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, i32, [0 x i32] }
%struct.GLDLayeredBufferstate = type { %union.GLSBuffer, %union.GLSBuffer, [8 x %union.GLSBuffer] }
%struct.GLDLight = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLDPointLineLimits, float, float, float, float, float, %struct.GLDPointLineLimits, float, %struct.GLDPointLineLimits, float, %struct.GLDPointLineLimits, float, float, float, float, float }
%struct.GLDLightModel = type { %struct.GLTColor4, [8 x %struct.GLDLight], [2 x %struct.GLDMaterial], i32, i16, i16, i16, i8, i8, i8, i8, i8, i8 }
%struct.GLDLightProduct = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4 }
%struct.GLDLineMode = type { float, i32, i16, i16, i8, i8, i8, i8 }
%struct.GLDLogicOp = type { i16, i8, i8 }
%struct.GLDMaskMode = type { i32, [3 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.GLDMaterial = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, float, float, float, float, [8 x %struct.GLDLightProduct], %struct.GLTColor4, [8 x i32] }
%struct.GLDMinmax = type { %struct.GLDMinmaxTable*, i16, i8, i8, [0 x i32] }
%struct.GLDMinmaxTable = type { %struct.GLTColor4, %struct.GLTColor4 }
%struct.GLDMipmaplevel = type { [4 x i32], [4 x i32], [4 x float], [4 x i32], i32, i32, float*, i8*, i16, i16, i16, i16, [2 x float] }
%struct.GLDMultisample = type { float, [1 x i32], [0 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.GLDMultisamplePositions = type { float, float }
%struct.GLDPipelineProgramData = type { i16, i8, i8, i32, %union.PPStreamToken*, i64, %struct.GLTColor4*, i32, [0 x i32] }
%struct.GLDPipelineProgramLimits = type { i32, i16, i16, i32, i16, i16, i32, i32 }
%struct.GLDPipelineProgramRec = type { %struct.GLDPipelineProgramData*, %union.PPStreamToken*, %struct.GLDContextRec*, %struct.GLVMProgramData*, i32, i32 }
%struct.GLDPipelineProgramState = type { i8, i8, i8, i8, [0 x i32], %struct.GLTColor4* }
%struct.GLDPixelFormatInfo = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.GLDPixelMap = type { i32*, float*, float*, float*, float*, float*, float*, float*, float*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.GLDPixelMode = type { float, float, %struct.GLDPixelStore, %struct.GLDPixelTransfer, %struct.GLDPixelMap, %struct.GLDImagingSubset, i32, [0 x i32] }
%struct.GLDPixelPack = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
%struct.GLDPixelStore = type { %struct.GLDPixelPack, %struct.GLDPixelPack }
%struct.GLDPixelTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32 }
%struct.GLDPluginBufferData = type { i32 }
%struct.GLDPluginFramebufferData = type { [10 x %struct.GLDTextureRec*], i8, i8, i8, i8 }
%struct.GLDPluginProgramData = type { [3 x %struct.GLDPipelineProgramRec*], %struct.GLDBufferRec**, i32, [0 x i32] }
%struct.GLDPluginState = type { [16 x [11 x %struct.GLDTextureRec*]], [3 x %struct.GLDTextureRec*], [3 x %struct.GLDPipelineProgramRec*], [3 x %struct.GLDPipelineProgramRec*], %struct.GLDProgramRec*, %struct.GLDVertexArrayRec*, [16 x %struct.GLDBufferRec*], %struct.GLDFramebufferRec*, %struct.GLDFramebufferRec*, [6 x %struct.GLDQueryRec*], [64 x %struct.GLDBufferRec*] }
%struct.GLDPluginTextureState = type { %struct.GLDBufferRec*, [6 x i16], i8, i8, i16 }
%struct.GLDPointLineLimits = type { float, float, float }
%struct.GLDPointMode = type { float, float, float, float, %struct.GLDPointLineLimits, float, i8, i8, i8, i8, i16, i16, i32, i16, i16 }
%struct.GLDPolygonMode = type { [128 x i8], float, float, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.GLDPolygonOffset = type { float, float }
%struct.GLDPrimitiveRestart = type { i8, i8, i8, i8, i32 }
%struct.GLDProgramData = type { i32, i32, i32, i32, %union.PPStreamToken*, i32*, i32, i32, i32, i32, i8, i8, i8, i8, i32, [64 x i32] }
%struct.GLDProgramLimits = type { i32, i32, i32, i32, i32, i16, i16 }
%struct.GLDProgramRec = type { %struct.GLDProgramData*, %struct.GLDPluginProgramData*, i32, [0 x i32] }
%struct.GLDProgramState = type { i8, i8, i8, i8 }
%struct.GLDQueryRec = type { i64, i64, i64 }
%struct.GLDQueryState = type { i16, i16 }
%struct.GLDReadBufferstate = type { %struct.GLTDimensions, %struct.GLTDimensions, %union.GLSBuffer, %union.GLSBuffer, %union.GLSBuffer, %union.GLSBuffer }
%struct.GLDRect = type { i32, i32, i32, i32, i32, i32 }
%struct.GLDRenderDispatch = type { void (%struct.GLDContextRec*, i32, float)*, void (%struct.GLDContextRec*, i32)*, i32 (%struct.GLDContextRec*, %struct.GLDMultisamplePositions*, i32, i32, i32, i32, i8*, i32, %struct.GLDBufferRec*)*, void (%struct.GLDContextRec*, %struct.GLDMultisamplePositions*, i32, i32, i32, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex*, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex**, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex**, i32, i32)*, void (%struct.GLDContextRec*, %struct.GLDVertex**, i32, i32)*, i8* (%struct.GLDContextRec*, i32, i32*)*, void (%struct.GLDContextRec*, i32, i32, i32)*, i8* (%struct.GLDContextRec*, i32, i32, i32, i32, i32)*, void (%struct.GLDContextRec*, i32, i32, i32, i32, i32, i8*)*, void (%struct.GLDContextRec*)*, void (%struct.GLDContextRec*)*, void (%struct.GLDContextRec*)*, i32 (%struct.GLDContextRec*, i32, i32, i32, i32, i32, i8*, %struct.GLTColor4*, i32)*, i32 (%struct.GLDContextRec*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* }
%struct.GLDRenderFeatures = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.GLDScissorTest = type { %struct.GLTFixedColor4, i8, i8, i8, i8 }
%struct.GLDSeamlessCubemap = type { i8, i8, i16 }
%struct.GLDSharedData = type {}
%struct.GLDSharedRec = type { %struct.pthread_mutex_t, %struct.GLDSharedData*, %struct.GLGProcessor, i32, i16, i16, i8, i8, i8, i8, [0 x i32] }
%struct.GLDState = type <{ i16, i16, i16, i16, i32, i32, [256 x %struct.GLTColor4], [128 x %struct.GLTColor4], %struct.GLDCurrent, %struct.GLDViewport, %struct.GLDTransform, %struct.GLDLightModel, %struct.GLDActiveTextureTargets, %struct.GLDAlphaTest, %struct.GLDBlendMode, %struct.GLDClearColor, %struct.GLDColorBuffer, %struct.GLDDepthTest, %struct.GLDArrayRange, %struct.GLDFogMode, %struct.GLDHintMode, %struct.GLDLineMode, %struct.GLDLogicOp, %struct.GLDMaskMode, %struct.GLDPixelMode, %struct.GLDPointMode, %struct.GLDPolygonMode, %struct.GLDScissorTest, i32, %struct.GLDStencilTest, [8 x %struct.GLDTextureMode], [16 x %struct.GLDTextureImageMode], [8 x %struct.GLDTextureCoordGen], %struct.GLDClipPlane, %struct.GLDMultisample, %struct.GLDArrayRange, %struct.GLDArrayRange, [3 x %struct.GLDPipelineProgramState], %struct.GLDArrayRange, %struct.GLDTransformFeedback, %struct.GLDUniformBuffer, i32*, %struct.GLDFixedFunction, i32, %struct.GLDQueryState, %struct.GLDSeamlessCubemap, %struct.GLDPrimitiveRestart, [2 x i32] }>
%struct.GLDStencilTest = type { [3 x %1], i32, [4 x i8] }
%struct.GLDStippleData = type { i32, i16, i16, [32 x [32 x i8]] }
%struct.GLDTextureCoordGen = type { %2, %2, %2, %2, i8, i8, i8, i8 }
%struct.GLDTextureGeomState = type { i16, i16, i16, i16, i16, i8, i8, i8, i8, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, [6 x i16], [6 x i16] }
%struct.GLDTextureImageMode = type { float }
%struct.GLDTextureLevel = type { i32, i32, i16, i16, i16, i8, i8, i16, i16, i16, i16, i8* }
%struct.GLDTextureLimits = type { float, float, i16, i16, i16, i16, i16, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i32, i32 }
%struct.GLDTextureMode = type { %struct.GLTColor4, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, float, float }
%struct.GLDTextureParamState = type { i16, i16, i16, i16, i16, i16, %struct.GLTColor4, float, float, float, float, i16, i16, i16, i16, float, i16, i16, i32, i8* }
%struct.GLDTextureRec = type { [4 x float], %struct.GLDTextureState*, %struct.GLDPluginTextureState*, %struct.GLDMipmaplevel*, %struct.GLDMipmaplevel*, float, float, float, float, i8, i8, i8, i8, i16, i16, i16, i16, i32, float, [0 x i32], [2 x %union.PPStreamToken] }
%struct.GLDTextureState = type { i16, i8, i8, i16, i16, float, i32, %struct.GLISWRSurface*, %struct.GLDTextureParamState, %struct.GLDTextureGeomState, i16, i16, i8*, %struct.GLDTextureLevel, [1 x [15 x %struct.GLDTextureLevel]] }
%struct.GLDTransform = type <{ [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, float, i8, i8, i8, i8, i32, i32, i32, i16, i16, i8, i8, i8, i8, i32 }>
%struct.GLDTransformFeedback = type { i8, i8, i16, [0 x i32], [16 x i32], [16 x i32] }
%struct.GLDTransformFeedbackLimits = type { i32, i32, i32, i32, i32 }
%struct.GLDUniformBuffer = type { [64 x %struct.GLTDimensions] }
%struct.GLDVertex = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLDPointLineLimits, float, %struct.GLTColor4, float, i8, i8, i8, i8, float, float, i32, i32, i32, [4 x i8], [4 x float], [2 x %struct.GLDMaterial*], [2 x i32], [16 x %struct.GLTColor4] }
%struct.GLDVertexArrayRec = type opaque
%struct.GLDVertexBlend = type { i8, i8, i8, i8 }
%struct.GLDVertexDescriptor = type { i8, i8, i8, i8, [0 x i32] }
%struct.GLDVertexProgramLimits = type { i16, i16, i32, i32, i16, i16 }
%struct.GLDViewport = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, double, double, i32, i32, i32, i32, float, float, float, float }
%struct.GLGColorTable = type { i32, i32, i32, i8* }
%struct.GLGOperation = type { i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, %struct.GLGColorTable, %struct.GLGColorTable, %struct.GLGColorTable }
%struct.GLGProcessor = type { void (%struct.GLDPixelMode*, %struct.GLGOperation*, %struct._GLGProcessorData*, %union._GLGFunctionKey*)*, %struct._GLVMFunction*, %union._GLGFunctionKey*, %struct._GLGProcessorData* }
%struct.GLISWRSurface = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i8*, [4 x i8*], i32 }
%struct.GLSGenericRec = type { %struct.GLTDimensions, %struct.GLTDimensions, i32, i32, %union.GLSDrawable, i8*, i8*, i8*, i8*, i8*, [4 x i8*], i32, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
%struct.GLSOffScreenRec = type { %struct.GLTDimensions, %struct.GLTDimensions, i32, i32, %union.GLSDrawable, i8*, i8*, i8*, i8*, i8*, [4 x i8*], i32, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, %struct.GLDDrawableOffscreen }
%struct.GLSSWRSurfaceRec = type { %struct.GLTDimensions, %struct.GLTDimensions, i32, i32, %union.GLSDrawable, i8*, i8*, i8*, i8*, i8*, [4 x i8*], i32, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, %struct.GLISWRSurface*, i32, i32 }
%struct.GLSWindowRec = type { %struct.GLTDimensions, %struct.GLTDimensions, i32, i32, %union.GLSDrawable, i8*, i8*, i8*, i8*, i8*, [4 x i8*], i32, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, %struct.GLDDrawableWindow, i32, i32, [0 x i32], i8*, i8* }
%struct.GLTColor3 = type { float, float, float }
%struct.GLTColor4 = type { float, float, float, float }
%struct.GLTCoord2 = type { float, float }
%struct.GLTCoord3 = type { float, float, float }
%struct.GLTCoord4 = type { float, float, float, float }
%struct.GLTDimensions = type { i32, i32 }
%struct.GLTFixedColor4 = type { i32, i32, i32, i32 }
%struct.GLTPlane = type { float, float, float, float }
%struct.GLTRectangle = type { i32, i32, i32, i32 }
%struct.GLTTexCoord4 = type { float, float, float, float }
%struct.GLVMFPContext = type { float, i32, i32, i32, i32, [3 x i32] }
%struct.GLVMFragmentAttribRec = type opaque
%struct.GLVMProgramData = type { %struct._GLVMFunction*, %struct.GLVMProgramData*, %struct.GLVMProgramData*, [42 x i32], [64 x i32], i32, i32, i32, [0 x i32] }
%struct.GLVMTextures = type { [16 x %struct.GLDTextureRec*] }
%struct._GLGProcessorData = type opaque
%struct._GLVMConstants = type opaque
%struct._GLVMFunction = type opaque
%struct.anon = type { float, float, float, float }
%struct.mach_timebase_info_data_t = type { i32, i32 }
%struct.pthread_mutex_t = type { i32, [40 x i8] }
%union.GLSBuffer = type { i8* }
%union.GLSDrawable = type { %struct.GLSWindowRec* }
%union.PPStreamToken = type { %0 }
%union._GLGFunctionKey = type opaque
%union.anon = type { %struct.GLTDimensions }

declare i16 @_OSSwapInt16(i16 zeroext %_data) nounwind inlinehint ssp

declare i32 @_OSSwapInt32(i32 %_data) nounwind inlinehint ssp

declare i32 @llvm.bswap.i32(i32) nounwind readnone

define void @gldAccumReturn(%struct.GLDContextRec* %ctx, %struct.GLDState* %state, float %value) nounwind ssp {
entry:
  %ctx_addr = alloca %struct.GLDContextRec*       ; <%struct.GLDContextRec**> [#uses=99]
  %state_addr = alloca %struct.GLDState*          ; <%struct.GLDState**> [#uses=21]
  %value_addr = alloca float                      ; <float*> [#uses=33]
  %iftmp.210 = alloca float                       ; <float*> [#uses=3]
  %iftmp.209 = alloca float                       ; <float*> [#uses=3]
  %iftmp.208 = alloca float                       ; <float*> [#uses=3]
  %iftmp.207 = alloca float                       ; <float*> [#uses=3]
  %iftmp.206 = alloca float                       ; <float*> [#uses=3]
  %iftmp.205 = alloca float                       ; <float*> [#uses=3]
  %iftmp.204 = alloca float                       ; <float*> [#uses=3]
  %iftmp.203 = alloca float                       ; <float*> [#uses=3]
  %iftmp.202 = alloca float                       ; <float*> [#uses=3]
  %iftmp.201 = alloca float                       ; <float*> [#uses=3]
  %iftmp.200 = alloca float                       ; <float*> [#uses=3]
  %iftmp.199 = alloca float                       ; <float*> [#uses=3]
  %iftmp.198 = alloca float                       ; <float*> [#uses=3]
  %iftmp.197 = alloca float                       ; <float*> [#uses=3]
  %iftmp.196 = alloca float                       ; <float*> [#uses=3]
  %iftmp.195 = alloca float                       ; <float*> [#uses=3]
  %iftmp.192 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.190 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.189 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.188 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.187 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.186 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.185 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.184 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.183 = alloca i32                         ; <i32*> [#uses=3]
  %iftmp.182 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.181 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.180 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.179 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.178 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.177 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.176 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.175 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.174 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.173 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.172 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.171 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.170 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.169 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.168 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.167 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.164 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.163 = alloca float                       ; <float*> [#uses=3]
  %iftmp.162 = alloca float                       ; <float*> [#uses=3]
  %iftmp.161 = alloca float                       ; <float*> [#uses=3]
  %iftmp.160 = alloca float                       ; <float*> [#uses=3]
  %iftmp.159 = alloca float                       ; <float*> [#uses=3]
  %iftmp.158 = alloca float                       ; <float*> [#uses=3]
  %iftmp.157 = alloca float                       ; <float*> [#uses=3]
  %iftmp.156 = alloca float                       ; <float*> [#uses=3]
  %iftmp.155 = alloca float                       ; <float*> [#uses=3]
  %iftmp.154 = alloca float                       ; <float*> [#uses=3]
  %iftmp.153 = alloca float                       ; <float*> [#uses=3]
  %iftmp.152 = alloca float                       ; <float*> [#uses=3]
  %iftmp.151 = alloca float                       ; <float*> [#uses=3]
  %iftmp.150 = alloca float                       ; <float*> [#uses=3]
  %iftmp.149 = alloca float                       ; <float*> [#uses=3]
  %iftmp.148 = alloca float                       ; <float*> [#uses=3]
  %iftmp.145 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.144 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.141 = alloca i8                          ; <i8*> [#uses=3]
  %iftmp.140 = alloca i32                         ; <i32*> [#uses=3]
  %accum = alloca double*                         ; <double**> [#uses=56]
  %accum_end = alloca double*                     ; <double**> [#uses=9]
  %y = alloca i32                                 ; <i32*> [#uses=12]
  %yl = alloca i32                                ; <i32*> [#uses=4]
  %cw4 = alloca i32                               ; <i32*> [#uses=4]
  %cx = alloca i32                                ; <i32*> [#uses=2]
  %cy = alloca i32                                ; <i32*> [#uses=7]
  %ch = alloca i32                                ; <i32*> [#uses=3]
  %cw = alloca i32                                ; <i32*> [#uses=3]
  %offset = alloca i32                            ; <i32*> [#uses=19]
  %y_inc = alloca i32                             ; <i32*> [#uses=6]
  %swap = alloca i8                               ; <i8*> [#uses=6]
  %draw_buffer = alloca i32                       ; <i32*> [#uses=21]
  %all_bits_mask = alloca i32                     ; <i32*> [#uses=9]
  %color_mask_enabled = alloca i8                 ; <i8*> [#uses=4]
  %color_ptr = alloca i16*                        ; <i16**> [#uses=8]
  %thirtyOne = alloca float                       ; <float*> [#uses=23]
  %start_offset = alloca i32                      ; <i32*> [#uses=2]
  %cur_draw_buffer_mask_bit = alloca i32          ; <i32*> [#uses=5]
  %r = alloca float                               ; <float*> [#uses=12]
  %g = alloca float                               ; <float*> [#uses=12]
  %b = alloca float                               ; <float*> [#uses=12]
  %a = alloca float                               ; <float*> [#uses=12]
  %pixl = alloca i16                              ; <i16*> [#uses=24]
  %color_ptr111 = alloca i8*                      ; <i8**> [#uses=14]
  %twoFiftyFive = alloca float                    ; <float*> [#uses=17]
  %start_offset112 = alloca i32                   ; <i32*> [#uses=2]
  %cur_draw_buffer_mask_bit115 = alloca i32       ; <i32*> [#uses=9]
  %r119 = alloca float                            ; <float*> [#uses=7]
  %g120 = alloca float                            ; <float*> [#uses=7]
  %b121 = alloca float                            ; <float*> [#uses=7]
  %a122 = alloca float                            ; <float*> [#uses=7]
  %r193 = alloca i32                              ; <i32*> [#uses=2]
  %g194 = alloca i32                              ; <i32*> [#uses=2]
  %b195 = alloca i32                              ; <i32*> [#uses=2]
  %a196 = alloca i32                              ; <i32*> [#uses=2]
  %color = alloca i32                             ; <i32*> [#uses=4]
  %color_ptr235 = alloca float*                   ; <float**> [#uses=13]
  %start_offset236 = alloca i32                   ; <i32*> [#uses=2]
  %cur_draw_buffer_mask_bit239 = alloca i32       ; <i32*> [#uses=5]
  %r243 = alloca float                            ; <float*> [#uses=4]
  %g244 = alloca float                            ; <float*> [#uses=4]
  %b245 = alloca float                            ; <float*> [#uses=4]
  %a246 = alloca float                            ; <float*> [#uses=4]
  %r283 = alloca float                            ; <float*> [#uses=4]
  %g284 = alloca float                            ; <float*> [#uses=4]
  %b285 = alloca float                            ; <float*> [#uses=4]
  %a286 = alloca float                            ; <float*> [#uses=4]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store %struct.GLDContextRec* %ctx, %struct.GLDContextRec** %ctx_addr
  store %struct.GLDState* %state, %struct.GLDState** %state_addr
  store float %value, float* %value_addr
  %0 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1 = getelementptr inbounds %struct.GLDContextRec* %0, i32 0, i32 51 ; <%union.GLSDrawable*> [#uses=1]
  %2 = getelementptr inbounds %union.GLSDrawable* %1, i32 0, i32 0 ; <%struct.GLSWindowRec**> [#uses=1]
  %3 = bitcast %struct.GLSWindowRec** %2 to %struct.GLSGenericRec** ; <%struct.GLSGenericRec**> [#uses=1]
  %4 = load %struct.GLSGenericRec** %3, align 4   ; <%struct.GLSGenericRec*> [#uses=1]
  %5 = getelementptr inbounds %struct.GLSGenericRec* %4, i32 0, i32 16 ; <i8*> [#uses=1]
  %6 = load i8* %5, align 4                       ; <i8> [#uses=1]
  store i8 %6, i8* %swap, align 1
  store i32 255, i32* %all_bits_mask, align 4
  %7 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %8 = getelementptr inbounds %struct.GLDState* %7, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %9 = getelementptr inbounds %struct.GLDMaskMode* %8, i32 0, i32 5 ; <i8*> [#uses=1]
  %10 = load i8* %9, align 1                      ; <i8> [#uses=1]
  %11 = zext i8 %10 to i32                        ; <i32> [#uses=1]
  %12 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %13 = and i32 %11, %12                          ; <i32> [#uses=1]
  %14 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %15 = icmp ne i32 %13, %14                      ; <i1> [#uses=1]
  %16 = zext i1 %15 to i8                         ; <i8> [#uses=1]
  %17 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %18 = getelementptr inbounds %struct.GLDState* %17, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %19 = getelementptr inbounds %struct.GLDMaskMode* %18, i32 0, i32 2 ; <i8*> [#uses=1]
  %20 = load i8* %19, align 16                    ; <i8> [#uses=1]
  %21 = zext i8 %20 to i32                        ; <i32> [#uses=1]
  %22 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %23 = and i32 %21, %22                          ; <i32> [#uses=1]
  %24 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %25 = icmp ne i32 %23, %24                      ; <i1> [#uses=1]
  %26 = zext i1 %25 to i8                         ; <i8> [#uses=1]
  %toBool = icmp ne i8 %16, 0                     ; <i1> [#uses=1]
  %toBool1 = icmp ne i8 %26, 0                    ; <i1> [#uses=1]
  %27 = or i1 %toBool, %toBool1                   ; <i1> [#uses=1]
  %28 = zext i1 %27 to i8                         ; <i8> [#uses=1]
  %29 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %30 = getelementptr inbounds %struct.GLDState* %29, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %31 = getelementptr inbounds %struct.GLDMaskMode* %30, i32 0, i32 4 ; <i8*> [#uses=1]
  %32 = load i8* %31, align 2                     ; <i8> [#uses=1]
  %33 = zext i8 %32 to i32                        ; <i32> [#uses=1]
  %34 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %35 = and i32 %33, %34                          ; <i32> [#uses=1]
  %36 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %37 = icmp ne i32 %35, %36                      ; <i1> [#uses=1]
  %38 = zext i1 %37 to i8                         ; <i8> [#uses=1]
  %toBool2 = icmp ne i8 %28, 0                    ; <i1> [#uses=1]
  %toBool3 = icmp ne i8 %38, 0                    ; <i1> [#uses=1]
  %39 = or i1 %toBool2, %toBool3                  ; <i1> [#uses=1]
  %40 = zext i1 %39 to i8                         ; <i8> [#uses=1]
  %41 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %42 = getelementptr inbounds %struct.GLDState* %41, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %43 = getelementptr inbounds %struct.GLDMaskMode* %42, i32 0, i32 3 ; <i8*> [#uses=1]
  %44 = load i8* %43, align 1                     ; <i8> [#uses=1]
  %45 = zext i8 %44 to i32                        ; <i32> [#uses=1]
  %46 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %47 = and i32 %45, %46                          ; <i32> [#uses=1]
  %48 = load i32* %all_bits_mask, align 4         ; <i32> [#uses=1]
  %49 = icmp ne i32 %47, %48                      ; <i1> [#uses=1]
  %50 = zext i1 %49 to i8                         ; <i8> [#uses=1]
  %toBool4 = icmp ne i8 %40, 0                    ; <i1> [#uses=1]
  %toBool5 = icmp ne i8 %50, 0                    ; <i1> [#uses=1]
  %51 = or i1 %toBool4, %toBool5                  ; <i1> [#uses=1]
  %52 = zext i1 %51 to i8                         ; <i8> [#uses=1]
  store i8 %52, i8* %color_mask_enabled, align 1
  %53 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %54 = getelementptr inbounds %struct.GLDContextRec* %53, i32 0, i32 55 ; <%struct.GLDRect*> [#uses=1]
  %55 = getelementptr inbounds %struct.GLDRect* %54, i32 0, i32 0 ; <i32*> [#uses=1]
  %56 = load i32* %55, align 4                    ; <i32> [#uses=1]
  store i32 %56, i32* %cx, align 4
  %57 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %58 = getelementptr inbounds %struct.GLDContextRec* %57, i32 0, i32 55 ; <%struct.GLDRect*> [#uses=1]
  %59 = getelementptr inbounds %struct.GLDRect* %58, i32 0, i32 1 ; <i32*> [#uses=1]
  %60 = load i32* %59, align 4                    ; <i32> [#uses=1]
  store i32 %60, i32* %cy, align 4
  %61 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %62 = getelementptr inbounds %struct.GLDContextRec* %61, i32 0, i32 55 ; <%struct.GLDRect*> [#uses=1]
  %63 = getelementptr inbounds %struct.GLDRect* %62, i32 0, i32 4 ; <i32*> [#uses=1]
  %64 = load i32* %63, align 4                    ; <i32> [#uses=1]
  store i32 %64, i32* %cw, align 4
  %65 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %66 = getelementptr inbounds %struct.GLDContextRec* %65, i32 0, i32 55 ; <%struct.GLDRect*> [#uses=1]
  %67 = getelementptr inbounds %struct.GLDRect* %66, i32 0, i32 5 ; <i32*> [#uses=1]
  %68 = load i32* %67, align 4                    ; <i32> [#uses=1]
  store i32 %68, i32* %ch, align 4
  %69 = load i32* %cw, align 4                    ; <i32> [#uses=1]
  %70 = icmp eq i32 %69, 0                        ; <i1> [#uses=1]
  br i1 %70, label %bb6, label %bb

bb:                                               ; preds = %entry
  %71 = load i32* %ch, align 4                    ; <i32> [#uses=1]
  %72 = icmp eq i32 %71, 0                        ; <i1> [#uses=1]
  br i1 %72, label %bb6, label %bb7

bb6:                                              ; preds = %bb, %entry
  br label %bb316

bb7:                                              ; preds = %bb
  %73 = load i32* %cw, align 4                    ; <i32> [#uses=1]
  %74 = mul i32 %73, 4                            ; <i32> [#uses=1]
  store i32 %74, i32* %cw4, align 4
  %75 = load i32* %cy, align 4                    ; <i32> [#uses=1]
  %76 = load i32* %ch, align 4                    ; <i32> [#uses=1]
  %77 = add i32 %75, %76                          ; <i32> [#uses=1]
  store i32 %77, i32* %yl, align 4
  %78 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %79 = getelementptr inbounds %struct.GLDContextRec* %78, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %80 = getelementptr inbounds %struct.GLDBufferstate* %79, i32 0, i32 1 ; <%struct.GLTDimensions*> [#uses=1]
  %81 = getelementptr inbounds %struct.GLTDimensions* %80, i32 0, i32 0 ; <i32*> [#uses=1]
  %82 = load i32* %81, align 4                    ; <i32> [#uses=1]
  %83 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %84 = getelementptr inbounds %struct.GLDContextRec* %83, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %85 = load %struct.GLDFramebufferRec** %84, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %86 = icmp ne %struct.GLDFramebufferRec* %85, null ; <i1> [#uses=1]
  br i1 %86, label %bb8, label %bb9

bb8:                                              ; preds = %bb7
  %87 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %88 = getelementptr inbounds %struct.GLDContextRec* %87, i32 0, i32 57 ; <%struct.GLDFormat*> [#uses=1]
  %89 = getelementptr inbounds %struct.GLDFormat* %88, i32 0, i32 12 ; <i8*> [#uses=1]
  %90 = load i8* %89, align 2                     ; <i8> [#uses=1]
  %91 = icmp ne i8 %90, 0                         ; <i1> [#uses=1]
  %92 = zext i1 %91 to i8                         ; <i8> [#uses=1]
  store i8 %92, i8* %iftmp.141, align 1
  br label %bb10

bb9:                                              ; preds = %bb7
  %93 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %94 = getelementptr inbounds %struct.GLDContextRec* %93, i32 0, i32 56 ; <%struct.GLDFormat*> [#uses=1]
  %95 = getelementptr inbounds %struct.GLDFormat* %94, i32 0, i32 12 ; <i8*> [#uses=1]
  %96 = load i8* %95, align 2                     ; <i8> [#uses=1]
  %97 = icmp ne i8 %96, 0                         ; <i1> [#uses=1]
  %98 = zext i1 %97 to i8                         ; <i8> [#uses=1]
  store i8 %98, i8* %iftmp.141, align 1
  br label %bb10

bb10:                                             ; preds = %bb9, %bb8
  %99 = load i8* %iftmp.141, align 1              ; <i8> [#uses=1]
  %toBool11 = icmp ne i8 %99, 0                   ; <i1> [#uses=1]
  br i1 %toBool11, label %bb12, label %bb13

bb12:                                             ; preds = %bb10
  %100 = load i32* %cy, align 4                   ; <i32> [#uses=1]
  store i32 %100, i32* %iftmp.140, align 4
  br label %bb14

bb13:                                             ; preds = %bb10
  %101 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %102 = getelementptr inbounds %struct.GLDContextRec* %101, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %103 = getelementptr inbounds %struct.GLDBufferstate* %102, i32 0, i32 1 ; <%struct.GLTDimensions*> [#uses=1]
  %104 = getelementptr inbounds %struct.GLTDimensions* %103, i32 0, i32 1 ; <i32*> [#uses=1]
  %105 = load i32* %104, align 4                  ; <i32> [#uses=1]
  %106 = sub nsw i32 %105, 1                      ; <i32> [#uses=1]
  %107 = load i32* %cy, align 4                   ; <i32> [#uses=1]
  %108 = sub nsw i32 %106, %107                   ; <i32> [#uses=1]
  store i32 %108, i32* %iftmp.140, align 4
  br label %bb14

bb14:                                             ; preds = %bb13, %bb12
  %109 = load i32* %iftmp.140, align 4            ; <i32> [#uses=1]
  %110 = mul nsw i32 %82, %109                    ; <i32> [#uses=1]
  %111 = load i32* %cx, align 4                   ; <i32> [#uses=1]
  %112 = add nsw i32 %110, %111                   ; <i32> [#uses=1]
  store i32 %112, i32* %offset, align 4
  %113 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %114 = getelementptr inbounds %struct.GLDContextRec* %113, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %115 = getelementptr inbounds %struct.GLDBufferstate* %114, i32 0, i32 1 ; <%struct.GLTDimensions*> [#uses=1]
  %116 = getelementptr inbounds %struct.GLTDimensions* %115, i32 0, i32 0 ; <i32*> [#uses=1]
  %117 = load i32* %116, align 4                  ; <i32> [#uses=1]
  store i32 %117, i32* %y_inc, align 4
  %118 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %119 = getelementptr inbounds %struct.GLDContextRec* %118, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %120 = load %struct.GLDFramebufferRec** %119, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %121 = icmp ne %struct.GLDFramebufferRec* %120, null ; <i1> [#uses=1]
  br i1 %121, label %bb15, label %bb16

bb15:                                             ; preds = %bb14
  %122 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %123 = getelementptr inbounds %struct.GLDContextRec* %122, i32 0, i32 57 ; <%struct.GLDFormat*> [#uses=1]
  %124 = getelementptr inbounds %struct.GLDFormat* %123, i32 0, i32 12 ; <i8*> [#uses=1]
  %125 = load i8* %124, align 2                   ; <i8> [#uses=1]
  %126 = icmp eq i8 %125, 0                       ; <i1> [#uses=1]
  %127 = zext i1 %126 to i8                       ; <i8> [#uses=1]
  store i8 %127, i8* %iftmp.144, align 1
  br label %bb17

bb16:                                             ; preds = %bb14
  %128 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %129 = getelementptr inbounds %struct.GLDContextRec* %128, i32 0, i32 56 ; <%struct.GLDFormat*> [#uses=1]
  %130 = getelementptr inbounds %struct.GLDFormat* %129, i32 0, i32 12 ; <i8*> [#uses=1]
  %131 = load i8* %130, align 2                   ; <i8> [#uses=1]
  %132 = icmp eq i8 %131, 0                       ; <i1> [#uses=1]
  %133 = zext i1 %132 to i8                       ; <i8> [#uses=1]
  store i8 %133, i8* %iftmp.144, align 1
  br label %bb17

bb17:                                             ; preds = %bb16, %bb15
  %134 = load i8* %iftmp.144, align 1             ; <i8> [#uses=1]
  %toBool18 = icmp ne i8 %134, 0                  ; <i1> [#uses=1]
  br i1 %toBool18, label %bb19, label %bb20

bb19:                                             ; preds = %bb17
  %135 = load i32* %y_inc, align 4                ; <i32> [#uses=1]
  %136 = sub i32 0, %135                          ; <i32> [#uses=1]
  store i32 %136, i32* %y_inc, align 4
  br label %bb20

bb20:                                             ; preds = %bb19, %bb17
  %137 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %138 = getelementptr inbounds %struct.GLDContextRec* %137, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %139 = load %struct.GLDFramebufferRec** %138, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %140 = icmp ne %struct.GLDFramebufferRec* %139, null ; <i1> [#uses=1]
  br i1 %140, label %bb21, label %bb22

bb21:                                             ; preds = %bb20
  %141 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %142 = getelementptr inbounds %struct.GLDContextRec* %141, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %143 = load %struct.GLDFramebufferRec** %142, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %144 = getelementptr inbounds %struct.GLDFramebufferRec* %143, i32 0, i32 2 ; <[10 x %struct.GLDFormat]*> [#uses=1]
  %145 = getelementptr inbounds [10 x %struct.GLDFormat]* %144, i32 0, i32 0 ; <%struct.GLDFormat*> [#uses=1]
  %146 = getelementptr inbounds %struct.GLDFormat* %145, i32 0, i32 7 ; <i32*> [#uses=1]
  %147 = load i32* %146, align 4                  ; <i32> [#uses=1]
  %148 = icmp eq i32 %147, 33638                  ; <i1> [#uses=1]
  %149 = zext i1 %148 to i8                       ; <i8> [#uses=1]
  store i8 %149, i8* %iftmp.145, align 1
  br label %bb23

bb22:                                             ; preds = %bb20
  %150 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %151 = getelementptr inbounds %struct.GLDContextRec* %150, i32 0, i32 56 ; <%struct.GLDFormat*> [#uses=1]
  %152 = getelementptr inbounds %struct.GLDFormat* %151, i32 0, i32 7 ; <i32*> [#uses=1]
  %153 = load i32* %152, align 4                  ; <i32> [#uses=1]
  %154 = icmp eq i32 %153, 33638                  ; <i1> [#uses=1]
  %155 = zext i1 %154 to i8                       ; <i8> [#uses=1]
  store i8 %155, i8* %iftmp.145, align 1
  br label %bb23

bb23:                                             ; preds = %bb22, %bb21
  %156 = load i8* %iftmp.145, align 1             ; <i8> [#uses=1]
  %toBool24 = icmp ne i8 %156, 0                  ; <i1> [#uses=1]
  br i1 %toBool24, label %bb25, label %bb105

bb25:                                             ; preds = %bb23
  %157 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %158 = getelementptr inbounds %struct.GLDContextRec* %157, i32 0, i32 6 ; <float*> [#uses=1]
  %159 = load float* %158, align 4                ; <float> [#uses=1]
  store float %159, float* %thirtyOne, align 4
  %160 = load i32* %offset, align 4               ; <i32> [#uses=1]
  store i32 %160, i32* %start_offset, align 4
  store i32 0, i32* %draw_buffer, align 4
  br label %bb104

bb26:                                             ; preds = %bb104
  %161 = load i32* %start_offset, align 4         ; <i32> [#uses=1]
  store i32 %161, i32* %offset, align 4
  %162 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %163 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %164 = getelementptr inbounds %struct.GLDContextRec* %163, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %165 = getelementptr inbounds %struct.GLDBufferstate* %164, i32 0, i32 12 ; <[8 x %union.GLSBuffer]*> [#uses=1]
  %166 = getelementptr inbounds [8 x %union.GLSBuffer]* %165, i32 0, i32 %162 ; <%union.GLSBuffer*> [#uses=1]
  %167 = getelementptr inbounds %union.GLSBuffer* %166, i32 0, i32 0 ; <i8**> [#uses=1]
  %168 = load i8** %167, align 4                  ; <i8*> [#uses=1]
  %169 = icmp ne i8* %168, null                   ; <i1> [#uses=1]
  br i1 %169, label %bb27, label %bb103

bb27:                                             ; preds = %bb26
  %170 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %171 = shl i32 1, %170                          ; <i32> [#uses=1]
  store i32 %171, i32* %cur_draw_buffer_mask_bit, align 4
  %172 = load i32* %cy, align 4                   ; <i32> [#uses=1]
  store i32 %172, i32* %y, align 4
  br label %bb102

bb28:                                             ; preds = %bb102
  %173 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %174 = getelementptr inbounds %struct.GLDContextRec* %173, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %175 = getelementptr inbounds %struct.GLDBufferstate* %174, i32 0, i32 11 ; <%union.GLSBuffer*> [#uses=1]
  %176 = getelementptr inbounds %union.GLSBuffer* %175, i32 0, i32 0 ; <i8**> [#uses=1]
  %177 = bitcast i8** %176 to double**            ; <double**> [#uses=1]
  %178 = load double** %177, align 4              ; <double*> [#uses=1]
  %179 = load i32* %offset, align 4               ; <i32> [#uses=1]
  %180 = mul i32 %179, 4                          ; <i32> [#uses=1]
  %181 = getelementptr inbounds double* %178, i32 %180 ; <double*> [#uses=1]
  store double* %181, double** %accum, align 4
  %182 = load double** %accum, align 4            ; <double*> [#uses=1]
  %183 = load i32* %cw4, align 4                  ; <i32> [#uses=1]
  %184 = getelementptr inbounds double* %182, i32 %183 ; <double*> [#uses=1]
  store double* %184, double** %accum_end, align 4
  %185 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %186 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %187 = getelementptr inbounds %struct.GLDContextRec* %186, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %188 = getelementptr inbounds %struct.GLDBufferstate* %187, i32 0, i32 12 ; <[8 x %union.GLSBuffer]*> [#uses=1]
  %189 = getelementptr inbounds [8 x %union.GLSBuffer]* %188, i32 0, i32 %185 ; <%union.GLSBuffer*> [#uses=1]
  %190 = getelementptr inbounds %union.GLSBuffer* %189, i32 0, i32 0 ; <i8**> [#uses=1]
  %191 = bitcast i8** %190 to i16**               ; <i16**> [#uses=1]
  %192 = load i16** %191, align 4                 ; <i16*> [#uses=1]
  %193 = load i32* %offset, align 4               ; <i32> [#uses=1]
  %194 = getelementptr inbounds i16* %192, i32 %193 ; <i16*> [#uses=1]
  store i16* %194, i16** %color_ptr, align 4
  %195 = load i8* %color_mask_enabled, align 1    ; <i8> [#uses=1]
  %196 = icmp ne i8 %195, 0                       ; <i1> [#uses=1]
  br i1 %196, label %bb29, label %bb70

bb29:                                             ; preds = %bb28
  br label %bb68

bb30:                                             ; preds = %bb68
  %197 = load double** %accum, align 4            ; <double*> [#uses=1]
  %198 = getelementptr inbounds double* %197, i32 3 ; <double*> [#uses=1]
  %199 = load double* %198, align 1               ; <double> [#uses=1]
  %200 = fptrunc double %199 to float             ; <float> [#uses=1]
  %201 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %202 = fmul float %200, %201                    ; <float> [#uses=1]
  store float %202, float* %a, align 4
  %203 = load double** %accum, align 4            ; <double*> [#uses=1]
  %204 = getelementptr inbounds double* %203, i32 0 ; <double*> [#uses=1]
  %205 = load double* %204, align 1               ; <double> [#uses=1]
  %206 = fptrunc double %205 to float             ; <float> [#uses=1]
  %207 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %208 = fmul float %206, %207                    ; <float> [#uses=1]
  %209 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %210 = fmul float %208, %209                    ; <float> [#uses=1]
  %211 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %212 = getelementptr inbounds %struct.GLDContextRec* %211, i32 0, i32 1 ; <float*> [#uses=1]
  %213 = load float* %212, align 4                ; <float> [#uses=1]
  %214 = fadd float %210, %213                    ; <float> [#uses=1]
  store float %214, float* %r, align 4
  %215 = load double** %accum, align 4            ; <double*> [#uses=1]
  %216 = getelementptr inbounds double* %215, i32 1 ; <double*> [#uses=1]
  %217 = load double* %216, align 1               ; <double> [#uses=1]
  %218 = fptrunc double %217 to float             ; <float> [#uses=1]
  %219 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %220 = fmul float %218, %219                    ; <float> [#uses=1]
  %221 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %222 = fmul float %220, %221                    ; <float> [#uses=1]
  %223 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %224 = getelementptr inbounds %struct.GLDContextRec* %223, i32 0, i32 1 ; <float*> [#uses=1]
  %225 = load float* %224, align 4                ; <float> [#uses=1]
  %226 = fadd float %222, %225                    ; <float> [#uses=1]
  store float %226, float* %g, align 4
  %227 = load double** %accum, align 4            ; <double*> [#uses=1]
  %228 = getelementptr inbounds double* %227, i32 2 ; <double*> [#uses=1]
  %229 = load double* %228, align 1               ; <double> [#uses=1]
  %230 = fptrunc double %229 to float             ; <float> [#uses=1]
  %231 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %232 = fmul float %230, %231                    ; <float> [#uses=1]
  %233 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %234 = fmul float %232, %233                    ; <float> [#uses=1]
  %235 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %236 = getelementptr inbounds %struct.GLDContextRec* %235, i32 0, i32 1 ; <float*> [#uses=1]
  %237 = load float* %236, align 4                ; <float> [#uses=1]
  %238 = fadd float %234, %237                    ; <float> [#uses=1]
  store float %238, float* %b, align 4
  %239 = load float* %r, align 4                  ; <float> [#uses=1]
  %240 = fcmp uge float %239, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %240, label %bb31, label %bb35

bb31:                                             ; preds = %bb30
  %241 = load float* %r, align 4                  ; <float> [#uses=1]
  %242 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %243 = fcmp ogt float %241, %242                ; <i1> [#uses=1]
  br i1 %243, label %bb32, label %bb33

bb32:                                             ; preds = %bb31
  %244 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %244, float* %iftmp.149, align 4
  br label %bb34

bb33:                                             ; preds = %bb31
  %245 = load float* %r, align 4                  ; <float> [#uses=1]
  store float %245, float* %iftmp.149, align 4
  br label %bb34

bb34:                                             ; preds = %bb33, %bb32
  %246 = load float* %iftmp.149, align 4          ; <float> [#uses=1]
  store float %246, float* %iftmp.148, align 4
  br label %bb36

bb35:                                             ; preds = %bb30
  store float 0.000000e+00, float* %iftmp.148, align 4
  br label %bb36

bb36:                                             ; preds = %bb35, %bb34
  %247 = load float* %iftmp.148, align 4          ; <float> [#uses=1]
  store float %247, float* %r, align 4
  %248 = load float* %g, align 4                  ; <float> [#uses=1]
  %249 = fcmp uge float %248, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %249, label %bb37, label %bb41

bb37:                                             ; preds = %bb36
  %250 = load float* %g, align 4                  ; <float> [#uses=1]
  %251 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %252 = fcmp ogt float %250, %251                ; <i1> [#uses=1]
  br i1 %252, label %bb38, label %bb39

bb38:                                             ; preds = %bb37
  %253 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %253, float* %iftmp.151, align 4
  br label %bb40

bb39:                                             ; preds = %bb37
  %254 = load float* %g, align 4                  ; <float> [#uses=1]
  store float %254, float* %iftmp.151, align 4
  br label %bb40

bb40:                                             ; preds = %bb39, %bb38
  %255 = load float* %iftmp.151, align 4          ; <float> [#uses=1]
  store float %255, float* %iftmp.150, align 4
  br label %bb42

bb41:                                             ; preds = %bb36
  store float 0.000000e+00, float* %iftmp.150, align 4
  br label %bb42

bb42:                                             ; preds = %bb41, %bb40
  %256 = load float* %iftmp.150, align 4          ; <float> [#uses=1]
  store float %256, float* %g, align 4
  %257 = load float* %b, align 4                  ; <float> [#uses=1]
  %258 = fcmp uge float %257, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %258, label %bb43, label %bb47

bb43:                                             ; preds = %bb42
  %259 = load float* %b, align 4                  ; <float> [#uses=1]
  %260 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %261 = fcmp ogt float %259, %260                ; <i1> [#uses=1]
  br i1 %261, label %bb44, label %bb45

bb44:                                             ; preds = %bb43
  %262 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %262, float* %iftmp.153, align 4
  br label %bb46

bb45:                                             ; preds = %bb43
  %263 = load float* %b, align 4                  ; <float> [#uses=1]
  store float %263, float* %iftmp.153, align 4
  br label %bb46

bb46:                                             ; preds = %bb45, %bb44
  %264 = load float* %iftmp.153, align 4          ; <float> [#uses=1]
  store float %264, float* %iftmp.152, align 4
  br label %bb48

bb47:                                             ; preds = %bb42
  store float 0.000000e+00, float* %iftmp.152, align 4
  br label %bb48

bb48:                                             ; preds = %bb47, %bb46
  %265 = load float* %iftmp.152, align 4          ; <float> [#uses=1]
  store float %265, float* %b, align 4
  %266 = load float* %a, align 4                  ; <float> [#uses=1]
  %267 = fcmp uge float %266, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %267, label %bb49, label %bb53

bb49:                                             ; preds = %bb48
  %268 = load float* %a, align 4                  ; <float> [#uses=1]
  %269 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %270 = fcmp ogt float %268, %269                ; <i1> [#uses=1]
  br i1 %270, label %bb50, label %bb51

bb50:                                             ; preds = %bb49
  %271 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %271, float* %iftmp.155, align 4
  br label %bb52

bb51:                                             ; preds = %bb49
  %272 = load float* %a, align 4                  ; <float> [#uses=1]
  store float %272, float* %iftmp.155, align 4
  br label %bb52

bb52:                                             ; preds = %bb51, %bb50
  %273 = load float* %iftmp.155, align 4          ; <float> [#uses=1]
  store float %273, float* %iftmp.154, align 4
  br label %bb54

bb53:                                             ; preds = %bb48
  store float 0.000000e+00, float* %iftmp.154, align 4
  br label %bb54

bb54:                                             ; preds = %bb53, %bb52
  %274 = load float* %iftmp.154, align 4          ; <float> [#uses=1]
  store float %274, float* %a, align 4
  %275 = load i16** %color_ptr, align 4           ; <i16*> [#uses=1]
  %276 = load i16* %275, align 2                  ; <i16> [#uses=1]
  store i16 %276, i16* %pixl, align 2
  %277 = load i8* %swap, align 1                  ; <i8> [#uses=1]
  %278 = icmp ne i8 %277, 0                       ; <i1> [#uses=1]
  br i1 %278, label %bb55, label %bb56

bb55:                                             ; preds = %bb54
  %279 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %280 = zext i16 %279 to i32                     ; <i32> [#uses=1]
  %281 = trunc i32 %280 to i16                    ; <i16> [#uses=1]
  %282 = call zeroext i16 @_OSSwapInt16(i16 zeroext %281) nounwind ; <i16> [#uses=1]
  store i16 %282, i16* %pixl, align 2
  br label %bb56

bb56:                                             ; preds = %bb55, %bb54
  %283 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %284 = getelementptr inbounds %struct.GLDState* %283, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %285 = getelementptr inbounds %struct.GLDMaskMode* %284, i32 0, i32 2 ; <i8*> [#uses=1]
  %286 = load i8* %285, align 16                  ; <i8> [#uses=1]
  %287 = zext i8 %286 to i32                      ; <i32> [#uses=1]
  %288 = load i32* %cur_draw_buffer_mask_bit, align 4 ; <i32> [#uses=1]
  %289 = and i32 %287, %288                       ; <i32> [#uses=1]
  %290 = icmp ne i32 %289, 0                      ; <i1> [#uses=1]
  br i1 %290, label %bb57, label %bb58

bb57:                                             ; preds = %bb56
  %291 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %292 = and i16 %291, -32257                     ; <i16> [#uses=1]
  %293 = load float* %r, align 4                  ; <float> [#uses=1]
  %294 = fptoui float %293 to i32                 ; <i32> [#uses=1]
  %295 = trunc i32 %294 to i16                    ; <i16> [#uses=1]
  %296 = shl i16 %295, 10                         ; <i16> [#uses=1]
  %297 = and i16 %296, 31744                      ; <i16> [#uses=1]
  %298 = or i16 %292, %297                        ; <i16> [#uses=1]
  store i16 %298, i16* %pixl, align 2
  br label %bb58

bb58:                                             ; preds = %bb57, %bb56
  %299 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %300 = getelementptr inbounds %struct.GLDState* %299, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %301 = getelementptr inbounds %struct.GLDMaskMode* %300, i32 0, i32 3 ; <i8*> [#uses=1]
  %302 = load i8* %301, align 1                   ; <i8> [#uses=1]
  %303 = zext i8 %302 to i32                      ; <i32> [#uses=1]
  %304 = load i32* %cur_draw_buffer_mask_bit, align 4 ; <i32> [#uses=1]
  %305 = and i32 %303, %304                       ; <i32> [#uses=1]
  %306 = icmp ne i32 %305, 0                      ; <i1> [#uses=1]
  br i1 %306, label %bb59, label %bb60

bb59:                                             ; preds = %bb58
  %307 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %308 = and i16 %307, -993                       ; <i16> [#uses=1]
  %309 = load float* %g, align 4                  ; <float> [#uses=1]
  %310 = fptoui float %309 to i32                 ; <i32> [#uses=1]
  %311 = trunc i32 %310 to i16                    ; <i16> [#uses=1]
  %312 = shl i16 %311, 5                          ; <i16> [#uses=1]
  %313 = and i16 %312, 992                        ; <i16> [#uses=1]
  %314 = or i16 %308, %313                        ; <i16> [#uses=1]
  store i16 %314, i16* %pixl, align 2
  br label %bb60

bb60:                                             ; preds = %bb59, %bb58
  %315 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %316 = getelementptr inbounds %struct.GLDState* %315, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %317 = getelementptr inbounds %struct.GLDMaskMode* %316, i32 0, i32 4 ; <i8*> [#uses=1]
  %318 = load i8* %317, align 2                   ; <i8> [#uses=1]
  %319 = zext i8 %318 to i32                      ; <i32> [#uses=1]
  %320 = load i32* %cur_draw_buffer_mask_bit, align 4 ; <i32> [#uses=1]
  %321 = and i32 %319, %320                       ; <i32> [#uses=1]
  %322 = icmp ne i32 %321, 0                      ; <i1> [#uses=1]
  br i1 %322, label %bb61, label %bb62

bb61:                                             ; preds = %bb60
  %323 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %324 = and i16 %323, -32                        ; <i16> [#uses=1]
  %325 = load float* %b, align 4                  ; <float> [#uses=1]
  %326 = fptoui float %325 to i32                 ; <i32> [#uses=1]
  %327 = trunc i32 %326 to i16                    ; <i16> [#uses=1]
  %328 = and i16 %327, 31                         ; <i16> [#uses=1]
  %329 = or i16 %324, %328                        ; <i16> [#uses=1]
  store i16 %329, i16* %pixl, align 2
  br label %bb62

bb62:                                             ; preds = %bb61, %bb60
  %330 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %331 = getelementptr inbounds %struct.GLDState* %330, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %332 = getelementptr inbounds %struct.GLDMaskMode* %331, i32 0, i32 5 ; <i8*> [#uses=1]
  %333 = load i8* %332, align 1                   ; <i8> [#uses=1]
  %334 = zext i8 %333 to i32                      ; <i32> [#uses=1]
  %335 = load i32* %cur_draw_buffer_mask_bit, align 4 ; <i32> [#uses=1]
  %336 = and i32 %334, %335                       ; <i32> [#uses=1]
  %337 = icmp ne i32 %336, 0                      ; <i1> [#uses=1]
  br i1 %337, label %bb63, label %bb65

bb63:                                             ; preds = %bb62
  %338 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %339 = getelementptr inbounds %struct.GLDContextRec* %338, i32 0, i32 1 ; <float*> [#uses=1]
  %340 = load float* %339, align 4                ; <float> [#uses=1]
  %341 = load float* %a, align 4                  ; <float> [#uses=1]
  %342 = fcmp olt float %340, %341                ; <i1> [#uses=1]
  br i1 %342, label %bb64, label %bb65

bb64:                                             ; preds = %bb63
  %343 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %344 = or i16 %343, -32768                      ; <i16> [#uses=1]
  store i16 %344, i16* %pixl, align 2
  br label %bb65

bb65:                                             ; preds = %bb64, %bb63, %bb62
  %345 = load i8* %swap, align 1                  ; <i8> [#uses=1]
  %346 = icmp ne i8 %345, 0                       ; <i1> [#uses=1]
  br i1 %346, label %bb66, label %bb67

bb66:                                             ; preds = %bb65
  %347 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %348 = zext i16 %347 to i32                     ; <i32> [#uses=1]
  %349 = trunc i32 %348 to i16                    ; <i16> [#uses=1]
  %350 = call zeroext i16 @_OSSwapInt16(i16 zeroext %349) nounwind ; <i16> [#uses=1]
  store i16 %350, i16* %pixl, align 2
  br label %bb67

bb67:                                             ; preds = %bb66, %bb65
  %351 = load i16** %color_ptr, align 4           ; <i16*> [#uses=1]
  %352 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  store i16 %352, i16* %351, align 2
  %353 = load double** %accum, align 4            ; <double*> [#uses=1]
  %354 = getelementptr inbounds double* %353, i32 4 ; <double*> [#uses=1]
  store double* %354, double** %accum, align 4
  %355 = load i16** %color_ptr, align 4           ; <i16*> [#uses=1]
  %356 = getelementptr inbounds i16* %355, i32 1  ; <i16*> [#uses=1]
  store i16* %356, i16** %color_ptr, align 4
  br label %bb68

bb68:                                             ; preds = %bb67, %bb29
  %357 = load double** %accum, align 4            ; <double*> [#uses=1]
  %358 = load double** %accum_end, align 4        ; <double*> [#uses=1]
  %359 = icmp ult double* %357, %358              ; <i1> [#uses=1]
  br i1 %359, label %bb30, label %bb69

bb69:                                             ; preds = %bb68
  br label %bb101

bb70:                                             ; preds = %bb28
  br label %bb100

bb71:                                             ; preds = %bb100
  %360 = load double** %accum, align 4            ; <double*> [#uses=1]
  %361 = getelementptr inbounds double* %360, i32 3 ; <double*> [#uses=1]
  %362 = load double* %361, align 1               ; <double> [#uses=1]
  %363 = fptrunc double %362 to float             ; <float> [#uses=1]
  %364 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %365 = fmul float %363, %364                    ; <float> [#uses=1]
  store float %365, float* %a, align 4
  %366 = load double** %accum, align 4            ; <double*> [#uses=1]
  %367 = getelementptr inbounds double* %366, i32 0 ; <double*> [#uses=1]
  %368 = load double* %367, align 1               ; <double> [#uses=1]
  %369 = fptrunc double %368 to float             ; <float> [#uses=1]
  %370 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %371 = fmul float %369, %370                    ; <float> [#uses=1]
  %372 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %373 = fmul float %371, %372                    ; <float> [#uses=1]
  %374 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %375 = getelementptr inbounds %struct.GLDContextRec* %374, i32 0, i32 1 ; <float*> [#uses=1]
  %376 = load float* %375, align 4                ; <float> [#uses=1]
  %377 = fadd float %373, %376                    ; <float> [#uses=1]
  store float %377, float* %r, align 4
  %378 = load double** %accum, align 4            ; <double*> [#uses=1]
  %379 = getelementptr inbounds double* %378, i32 1 ; <double*> [#uses=1]
  %380 = load double* %379, align 1               ; <double> [#uses=1]
  %381 = fptrunc double %380 to float             ; <float> [#uses=1]
  %382 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %383 = fmul float %381, %382                    ; <float> [#uses=1]
  %384 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %385 = fmul float %383, %384                    ; <float> [#uses=1]
  %386 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %387 = getelementptr inbounds %struct.GLDContextRec* %386, i32 0, i32 1 ; <float*> [#uses=1]
  %388 = load float* %387, align 4                ; <float> [#uses=1]
  %389 = fadd float %385, %388                    ; <float> [#uses=1]
  store float %389, float* %g, align 4
  %390 = load double** %accum, align 4            ; <double*> [#uses=1]
  %391 = getelementptr inbounds double* %390, i32 2 ; <double*> [#uses=1]
  %392 = load double* %391, align 1               ; <double> [#uses=1]
  %393 = fptrunc double %392 to float             ; <float> [#uses=1]
  %394 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %395 = fmul float %393, %394                    ; <float> [#uses=1]
  %396 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %397 = fmul float %395, %396                    ; <float> [#uses=1]
  %398 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %399 = getelementptr inbounds %struct.GLDContextRec* %398, i32 0, i32 1 ; <float*> [#uses=1]
  %400 = load float* %399, align 4                ; <float> [#uses=1]
  %401 = fadd float %397, %400                    ; <float> [#uses=1]
  store float %401, float* %b, align 4
  %402 = load float* %r, align 4                  ; <float> [#uses=1]
  %403 = fcmp uge float %402, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %403, label %bb72, label %bb76

bb72:                                             ; preds = %bb71
  %404 = load float* %r, align 4                  ; <float> [#uses=1]
  %405 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %406 = fcmp ogt float %404, %405                ; <i1> [#uses=1]
  br i1 %406, label %bb73, label %bb74

bb73:                                             ; preds = %bb72
  %407 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %407, float* %iftmp.157, align 4
  br label %bb75

bb74:                                             ; preds = %bb72
  %408 = load float* %r, align 4                  ; <float> [#uses=1]
  store float %408, float* %iftmp.157, align 4
  br label %bb75

bb75:                                             ; preds = %bb74, %bb73
  %409 = load float* %iftmp.157, align 4          ; <float> [#uses=1]
  store float %409, float* %iftmp.156, align 4
  br label %bb77

bb76:                                             ; preds = %bb71
  store float 0.000000e+00, float* %iftmp.156, align 4
  br label %bb77

bb77:                                             ; preds = %bb76, %bb75
  %410 = load float* %iftmp.156, align 4          ; <float> [#uses=1]
  store float %410, float* %r, align 4
  %411 = load float* %g, align 4                  ; <float> [#uses=1]
  %412 = fcmp uge float %411, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %412, label %bb78, label %bb82

bb78:                                             ; preds = %bb77
  %413 = load float* %g, align 4                  ; <float> [#uses=1]
  %414 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %415 = fcmp ogt float %413, %414                ; <i1> [#uses=1]
  br i1 %415, label %bb79, label %bb80

bb79:                                             ; preds = %bb78
  %416 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %416, float* %iftmp.159, align 4
  br label %bb81

bb80:                                             ; preds = %bb78
  %417 = load float* %g, align 4                  ; <float> [#uses=1]
  store float %417, float* %iftmp.159, align 4
  br label %bb81

bb81:                                             ; preds = %bb80, %bb79
  %418 = load float* %iftmp.159, align 4          ; <float> [#uses=1]
  store float %418, float* %iftmp.158, align 4
  br label %bb83

bb82:                                             ; preds = %bb77
  store float 0.000000e+00, float* %iftmp.158, align 4
  br label %bb83

bb83:                                             ; preds = %bb82, %bb81
  %419 = load float* %iftmp.158, align 4          ; <float> [#uses=1]
  store float %419, float* %g, align 4
  %420 = load float* %b, align 4                  ; <float> [#uses=1]
  %421 = fcmp uge float %420, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %421, label %bb84, label %bb88

bb84:                                             ; preds = %bb83
  %422 = load float* %b, align 4                  ; <float> [#uses=1]
  %423 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %424 = fcmp ogt float %422, %423                ; <i1> [#uses=1]
  br i1 %424, label %bb85, label %bb86

bb85:                                             ; preds = %bb84
  %425 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %425, float* %iftmp.161, align 4
  br label %bb87

bb86:                                             ; preds = %bb84
  %426 = load float* %b, align 4                  ; <float> [#uses=1]
  store float %426, float* %iftmp.161, align 4
  br label %bb87

bb87:                                             ; preds = %bb86, %bb85
  %427 = load float* %iftmp.161, align 4          ; <float> [#uses=1]
  store float %427, float* %iftmp.160, align 4
  br label %bb89

bb88:                                             ; preds = %bb83
  store float 0.000000e+00, float* %iftmp.160, align 4
  br label %bb89

bb89:                                             ; preds = %bb88, %bb87
  %428 = load float* %iftmp.160, align 4          ; <float> [#uses=1]
  store float %428, float* %b, align 4
  %429 = load float* %a, align 4                  ; <float> [#uses=1]
  %430 = fcmp uge float %429, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %430, label %bb90, label %bb94

bb90:                                             ; preds = %bb89
  %431 = load float* %a, align 4                  ; <float> [#uses=1]
  %432 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  %433 = fcmp ogt float %431, %432                ; <i1> [#uses=1]
  br i1 %433, label %bb91, label %bb92

bb91:                                             ; preds = %bb90
  %434 = load float* %thirtyOne, align 4          ; <float> [#uses=1]
  store float %434, float* %iftmp.163, align 4
  br label %bb93

bb92:                                             ; preds = %bb90
  %435 = load float* %a, align 4                  ; <float> [#uses=1]
  store float %435, float* %iftmp.163, align 4
  br label %bb93

bb93:                                             ; preds = %bb92, %bb91
  %436 = load float* %iftmp.163, align 4          ; <float> [#uses=1]
  store float %436, float* %iftmp.162, align 4
  br label %bb95

bb94:                                             ; preds = %bb89
  store float 0.000000e+00, float* %iftmp.162, align 4
  br label %bb95

bb95:                                             ; preds = %bb94, %bb93
  %437 = load float* %iftmp.162, align 4          ; <float> [#uses=1]
  store float %437, float* %a, align 4
  %438 = load float* %r, align 4                  ; <float> [#uses=1]
  %439 = fptoui float %438 to i32                 ; <i32> [#uses=1]
  %440 = trunc i32 %439 to i16                    ; <i16> [#uses=1]
  %441 = shl i16 %440, 10                         ; <i16> [#uses=1]
  %442 = and i16 %441, 31744                      ; <i16> [#uses=1]
  store i16 %442, i16* %pixl, align 2
  %443 = load float* %g, align 4                  ; <float> [#uses=1]
  %444 = fptoui float %443 to i32                 ; <i32> [#uses=1]
  %445 = trunc i32 %444 to i16                    ; <i16> [#uses=1]
  %446 = shl i16 %445, 5                          ; <i16> [#uses=1]
  %447 = and i16 %446, 992                        ; <i16> [#uses=1]
  %448 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %449 = or i16 %447, %448                        ; <i16> [#uses=1]
  store i16 %449, i16* %pixl, align 2
  %450 = load float* %b, align 4                  ; <float> [#uses=1]
  %451 = fptoui float %450 to i32                 ; <i32> [#uses=1]
  %452 = trunc i32 %451 to i16                    ; <i16> [#uses=1]
  %453 = and i16 %452, 31                         ; <i16> [#uses=1]
  %454 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %455 = or i16 %453, %454                        ; <i16> [#uses=1]
  store i16 %455, i16* %pixl, align 2
  %456 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %457 = getelementptr inbounds %struct.GLDContextRec* %456, i32 0, i32 1 ; <float*> [#uses=1]
  %458 = load float* %457, align 4                ; <float> [#uses=1]
  %459 = load float* %a, align 4                  ; <float> [#uses=1]
  %460 = fcmp olt float %458, %459                ; <i1> [#uses=1]
  br i1 %460, label %bb96, label %bb97

bb96:                                             ; preds = %bb95
  %461 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %462 = or i16 %461, -32768                      ; <i16> [#uses=1]
  store i16 %462, i16* %pixl, align 2
  br label %bb97

bb97:                                             ; preds = %bb96, %bb95
  %463 = load i8* %swap, align 1                  ; <i8> [#uses=1]
  %464 = icmp ne i8 %463, 0                       ; <i1> [#uses=1]
  br i1 %464, label %bb98, label %bb99

bb98:                                             ; preds = %bb97
  %465 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  %466 = zext i16 %465 to i32                     ; <i32> [#uses=1]
  %467 = trunc i32 %466 to i16                    ; <i16> [#uses=1]
  %468 = call zeroext i16 @_OSSwapInt16(i16 zeroext %467) nounwind ; <i16> [#uses=1]
  store i16 %468, i16* %pixl, align 2
  br label %bb99

bb99:                                             ; preds = %bb98, %bb97
  %469 = load i16** %color_ptr, align 4           ; <i16*> [#uses=1]
  %470 = load i16* %pixl, align 2                 ; <i16> [#uses=1]
  store i16 %470, i16* %469, align 2
  %471 = load double** %accum, align 4            ; <double*> [#uses=1]
  %472 = getelementptr inbounds double* %471, i32 4 ; <double*> [#uses=1]
  store double* %472, double** %accum, align 4
  %473 = load i16** %color_ptr, align 4           ; <i16*> [#uses=1]
  %474 = getelementptr inbounds i16* %473, i32 1  ; <i16*> [#uses=1]
  store i16* %474, i16** %color_ptr, align 4
  br label %bb100

bb100:                                            ; preds = %bb99, %bb70
  %475 = load double** %accum, align 4            ; <double*> [#uses=1]
  %476 = load double** %accum_end, align 4        ; <double*> [#uses=1]
  %477 = icmp ult double* %475, %476              ; <i1> [#uses=1]
  br i1 %477, label %bb71, label %bb101

bb101:                                            ; preds = %bb100, %bb69
  %478 = load i32* %y, align 4                    ; <i32> [#uses=1]
  %479 = add i32 %478, 1                          ; <i32> [#uses=1]
  store i32 %479, i32* %y, align 4
  %480 = load i32* %offset, align 4               ; <i32> [#uses=1]
  %481 = load i32* %y_inc, align 4                ; <i32> [#uses=1]
  %482 = add i32 %480, %481                       ; <i32> [#uses=1]
  store i32 %482, i32* %offset, align 4
  br label %bb102

bb102:                                            ; preds = %bb101, %bb27
  %483 = load i32* %y, align 4                    ; <i32> [#uses=1]
  %484 = load i32* %yl, align 4                   ; <i32> [#uses=1]
  %485 = icmp ult i32 %483, %484                  ; <i1> [#uses=1]
  br i1 %485, label %bb28, label %bb103

bb103:                                            ; preds = %bb102, %bb26
  %486 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %487 = add nsw i32 %486, 1                      ; <i32> [#uses=1]
  store i32 %487, i32* %draw_buffer, align 4
  br label %bb104

bb104:                                            ; preds = %bb103, %bb25
  %488 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %489 = icmp sle i32 %488, 7                     ; <i1> [#uses=1]
  br i1 %489, label %bb26, label %bb105

bb105:                                            ; preds = %bb104, %bb23
  %490 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %491 = getelementptr inbounds %struct.GLDContextRec* %490, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %492 = load %struct.GLDFramebufferRec** %491, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %493 = icmp ne %struct.GLDFramebufferRec* %492, null ; <i1> [#uses=1]
  br i1 %493, label %bb106, label %bb107

bb106:                                            ; preds = %bb105
  %494 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %495 = getelementptr inbounds %struct.GLDContextRec* %494, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %496 = load %struct.GLDFramebufferRec** %495, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %497 = getelementptr inbounds %struct.GLDFramebufferRec* %496, i32 0, i32 2 ; <[10 x %struct.GLDFormat]*> [#uses=1]
  %498 = getelementptr inbounds [10 x %struct.GLDFormat]* %497, i32 0, i32 0 ; <%struct.GLDFormat*> [#uses=1]
  %499 = getelementptr inbounds %struct.GLDFormat* %498, i32 0, i32 7 ; <i32*> [#uses=1]
  %500 = load i32* %499, align 4                  ; <i32> [#uses=1]
  %501 = icmp eq i32 %500, 33639                  ; <i1> [#uses=1]
  %502 = zext i1 %501 to i8                       ; <i8> [#uses=1]
  store i8 %502, i8* %iftmp.164, align 1
  br label %bb108

bb107:                                            ; preds = %bb105
  %503 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %504 = getelementptr inbounds %struct.GLDContextRec* %503, i32 0, i32 56 ; <%struct.GLDFormat*> [#uses=1]
  %505 = getelementptr inbounds %struct.GLDFormat* %504, i32 0, i32 7 ; <i32*> [#uses=1]
  %506 = load i32* %505, align 4                  ; <i32> [#uses=1]
  %507 = icmp eq i32 %506, 33639                  ; <i1> [#uses=1]
  %508 = zext i1 %507 to i8                       ; <i8> [#uses=1]
  store i8 %508, i8* %iftmp.164, align 1
  br label %bb108

bb108:                                            ; preds = %bb107, %bb106
  %509 = load i8* %iftmp.164, align 1             ; <i8> [#uses=1]
  %toBool109 = icmp ne i8 %509, 0                 ; <i1> [#uses=1]
  br i1 %toBool109, label %bb110, label %bb229

bb110:                                            ; preds = %bb108
  %510 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %511 = getelementptr inbounds %struct.GLDContextRec* %510, i32 0, i32 7 ; <float*> [#uses=1]
  %512 = load float* %511, align 4                ; <float> [#uses=1]
  store float %512, float* %twoFiftyFive, align 4
  %513 = load i32* %offset, align 4               ; <i32> [#uses=1]
  store i32 %513, i32* %start_offset112, align 4
  store i32 0, i32* %draw_buffer, align 4
  br label %bb227

bb113:                                            ; preds = %bb227
  %514 = load i32* %start_offset112, align 4      ; <i32> [#uses=1]
  store i32 %514, i32* %offset, align 4
  %515 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %516 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %517 = getelementptr inbounds %struct.GLDContextRec* %516, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %518 = getelementptr inbounds %struct.GLDBufferstate* %517, i32 0, i32 12 ; <[8 x %union.GLSBuffer]*> [#uses=1]
  %519 = getelementptr inbounds [8 x %union.GLSBuffer]* %518, i32 0, i32 %515 ; <%union.GLSBuffer*> [#uses=1]
  %520 = getelementptr inbounds %union.GLSBuffer* %519, i32 0, i32 0 ; <i8**> [#uses=1]
  %521 = load i8** %520, align 4                  ; <i8*> [#uses=1]
  %522 = icmp ne i8* %521, null                   ; <i1> [#uses=1]
  br i1 %522, label %bb114, label %bb226

bb114:                                            ; preds = %bb113
  %523 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %524 = shl i32 1, %523                          ; <i32> [#uses=1]
  store i32 %524, i32* %cur_draw_buffer_mask_bit115, align 4
  %525 = load i32* %cy, align 4                   ; <i32> [#uses=1]
  store i32 %525, i32* %y, align 4
  br label %bb225

bb116:                                            ; preds = %bb225
  %526 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %527 = getelementptr inbounds %struct.GLDContextRec* %526, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %528 = getelementptr inbounds %struct.GLDBufferstate* %527, i32 0, i32 11 ; <%union.GLSBuffer*> [#uses=1]
  %529 = getelementptr inbounds %union.GLSBuffer* %528, i32 0, i32 0 ; <i8**> [#uses=1]
  %530 = bitcast i8** %529 to double**            ; <double**> [#uses=1]
  %531 = load double** %530, align 4              ; <double*> [#uses=1]
  %532 = load i32* %offset, align 4               ; <i32> [#uses=1]
  %533 = mul i32 %532, 4                          ; <i32> [#uses=1]
  %534 = getelementptr inbounds double* %531, i32 %533 ; <double*> [#uses=1]
  store double* %534, double** %accum, align 4
  %535 = load double** %accum, align 4            ; <double*> [#uses=1]
  %536 = load i32* %cw4, align 4                  ; <i32> [#uses=1]
  %537 = getelementptr inbounds double* %535, i32 %536 ; <double*> [#uses=1]
  store double* %537, double** %accum_end, align 4
  %538 = load i32* %draw_buffer, align 4          ; <i32> [#uses=1]
  %539 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %540 = getelementptr inbounds %struct.GLDContextRec* %539, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %541 = getelementptr inbounds %struct.GLDBufferstate* %540, i32 0, i32 12 ; <[8 x %union.GLSBuffer]*> [#uses=1]
  %542 = getelementptr inbounds [8 x %union.GLSBuffer]* %541, i32 0, i32 %538 ; <%union.GLSBuffer*> [#uses=1]
  %543 = getelementptr inbounds %union.GLSBuffer* %542, i32 0, i32 0 ; <i8**> [#uses=1]
  %544 = bitcast i8** %543 to i32**               ; <i32**> [#uses=1]
  %545 = load i32** %544, align 4                 ; <i32*> [#uses=1]
  %546 = load i32* %offset, align 4               ; <i32> [#uses=1]
  %547 = getelementptr inbounds i32* %545, i32 %546 ; <i32*> [#uses=1]
  %548 = bitcast i32* %547 to i8*                 ; <i8*> [#uses=1]
  store i8* %548, i8** %color_ptr111, align 4
  %549 = load i8* %color_mask_enabled, align 1    ; <i8> [#uses=1]
  %550 = icmp ne i8 %549, 0                       ; <i1> [#uses=1]
  br i1 %550, label %bb117, label %bb191

bb117:                                            ; preds = %bb116
  br label %bb189

bb118:                                            ; preds = %bb189
  %551 = load double** %accum, align 4            ; <double*> [#uses=1]
  %552 = getelementptr inbounds double* %551, i32 0 ; <double*> [#uses=1]
  %553 = load double* %552, align 1               ; <double> [#uses=1]
  %554 = fptrunc double %553 to float             ; <float> [#uses=1]
  %555 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %556 = fmul float %554, %555                    ; <float> [#uses=1]
  %557 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %558 = fmul float %556, %557                    ; <float> [#uses=1]
  %559 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %560 = getelementptr inbounds %struct.GLDContextRec* %559, i32 0, i32 1 ; <float*> [#uses=1]
  %561 = load float* %560, align 4                ; <float> [#uses=1]
  %562 = fadd float %558, %561                    ; <float> [#uses=1]
  store float %562, float* %r119, align 4
  %563 = load double** %accum, align 4            ; <double*> [#uses=1]
  %564 = getelementptr inbounds double* %563, i32 1 ; <double*> [#uses=1]
  %565 = load double* %564, align 1               ; <double> [#uses=1]
  %566 = fptrunc double %565 to float             ; <float> [#uses=1]
  %567 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %568 = fmul float %566, %567                    ; <float> [#uses=1]
  %569 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %570 = fmul float %568, %569                    ; <float> [#uses=1]
  %571 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %572 = getelementptr inbounds %struct.GLDContextRec* %571, i32 0, i32 1 ; <float*> [#uses=1]
  %573 = load float* %572, align 4                ; <float> [#uses=1]
  %574 = fadd float %570, %573                    ; <float> [#uses=1]
  store float %574, float* %g120, align 4
  %575 = load double** %accum, align 4            ; <double*> [#uses=1]
  %576 = getelementptr inbounds double* %575, i32 2 ; <double*> [#uses=1]
  %577 = load double* %576, align 1               ; <double> [#uses=1]
  %578 = fptrunc double %577 to float             ; <float> [#uses=1]
  %579 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %580 = fmul float %578, %579                    ; <float> [#uses=1]
  %581 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %582 = fmul float %580, %581                    ; <float> [#uses=1]
  %583 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %584 = getelementptr inbounds %struct.GLDContextRec* %583, i32 0, i32 1 ; <float*> [#uses=1]
  %585 = load float* %584, align 4                ; <float> [#uses=1]
  %586 = fadd float %582, %585                    ; <float> [#uses=1]
  store float %586, float* %b121, align 4
  %587 = load double** %accum, align 4            ; <double*> [#uses=1]
  %588 = getelementptr inbounds double* %587, i32 3 ; <double*> [#uses=1]
  %589 = load double* %588, align 1               ; <double> [#uses=1]
  %590 = fptrunc double %589 to float             ; <float> [#uses=1]
  %591 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %592 = fmul float %590, %591                    ; <float> [#uses=1]
  %593 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %594 = fmul float %592, %593                    ; <float> [#uses=1]
  %595 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %596 = getelementptr inbounds %struct.GLDContextRec* %595, i32 0, i32 1 ; <float*> [#uses=1]
  %597 = load float* %596, align 4                ; <float> [#uses=1]
  %598 = fadd float %594, %597                    ; <float> [#uses=1]
  store float %598, float* %a122, align 4
  %599 = load i8* %swap, align 1                  ; <i8> [#uses=1]
  %600 = icmp ne i8 %599, 0                       ; <i1> [#uses=1]
  br i1 %600, label %bb123, label %bb156

bb123:                                            ; preds = %bb118
  %601 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %602 = getelementptr inbounds %struct.GLDState* %601, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %603 = getelementptr inbounds %struct.GLDMaskMode* %602, i32 0, i32 5 ; <i8*> [#uses=1]
  %604 = load i8* %603, align 1                   ; <i8> [#uses=1]
  %605 = zext i8 %604 to i32                      ; <i32> [#uses=1]
  %606 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %607 = and i32 %605, %606                       ; <i32> [#uses=1]
  %608 = icmp ne i32 %607, 0                      ; <i1> [#uses=1]
  br i1 %608, label %bb124, label %bb131

bb124:                                            ; preds = %bb123
  %609 = load float* %a122, align 4               ; <float> [#uses=1]
  %610 = fcmp uge float %609, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %610, label %bb125, label %bb129

bb125:                                            ; preds = %bb124
  %611 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %612 = getelementptr inbounds %struct.GLDContextRec* %611, i32 0, i32 7 ; <float*> [#uses=1]
  %613 = load float* %612, align 4                ; <float> [#uses=1]
  %614 = load float* %a122, align 4               ; <float> [#uses=1]
  %615 = fcmp olt float %613, %614                ; <i1> [#uses=1]
  br i1 %615, label %bb126, label %bb127

bb126:                                            ; preds = %bb125
  %616 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %617 = getelementptr inbounds %struct.GLDContextRec* %616, i32 0, i32 7 ; <float*> [#uses=1]
  %618 = load float* %617, align 4                ; <float> [#uses=1]
  %619 = fptoui float %618 to i8                  ; <i8> [#uses=1]
  store i8 %619, i8* %iftmp.168, align 1
  br label %bb128

bb127:                                            ; preds = %bb125
  %620 = load float* %a122, align 4               ; <float> [#uses=1]
  %621 = fptoui float %620 to i8                  ; <i8> [#uses=1]
  store i8 %621, i8* %iftmp.168, align 1
  br label %bb128

bb128:                                            ; preds = %bb127, %bb126
  %622 = load i8* %iftmp.168, align 1             ; <i8> [#uses=1]
  store i8 %622, i8* %iftmp.167, align 1
  br label %bb130

bb129:                                            ; preds = %bb124
  store i8 0, i8* %iftmp.167, align 1
  br label %bb130

bb130:                                            ; preds = %bb129, %bb128
  %623 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %624 = getelementptr inbounds i8* %623, i32 0   ; <i8*> [#uses=1]
  %625 = load i8* %iftmp.167, align 1             ; <i8> [#uses=1]
  store i8 %625, i8* %624, align 1
  br label %bb131

bb131:                                            ; preds = %bb130, %bb123
  %626 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %627 = getelementptr inbounds %struct.GLDState* %626, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %628 = getelementptr inbounds %struct.GLDMaskMode* %627, i32 0, i32 2 ; <i8*> [#uses=1]
  %629 = load i8* %628, align 16                  ; <i8> [#uses=1]
  %630 = zext i8 %629 to i32                      ; <i32> [#uses=1]
  %631 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %632 = and i32 %630, %631                       ; <i32> [#uses=1]
  %633 = icmp ne i32 %632, 0                      ; <i1> [#uses=1]
  br i1 %633, label %bb132, label %bb139

bb132:                                            ; preds = %bb131
  %634 = load float* %r119, align 4               ; <float> [#uses=1]
  %635 = fcmp uge float %634, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %635, label %bb133, label %bb137

bb133:                                            ; preds = %bb132
  %636 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %637 = getelementptr inbounds %struct.GLDContextRec* %636, i32 0, i32 7 ; <float*> [#uses=1]
  %638 = load float* %637, align 4                ; <float> [#uses=1]
  %639 = load float* %r119, align 4               ; <float> [#uses=1]
  %640 = fcmp olt float %638, %639                ; <i1> [#uses=1]
  br i1 %640, label %bb134, label %bb135

bb134:                                            ; preds = %bb133
  %641 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %642 = getelementptr inbounds %struct.GLDContextRec* %641, i32 0, i32 7 ; <float*> [#uses=1]
  %643 = load float* %642, align 4                ; <float> [#uses=1]
  %644 = fptoui float %643 to i8                  ; <i8> [#uses=1]
  store i8 %644, i8* %iftmp.170, align 1
  br label %bb136

bb135:                                            ; preds = %bb133
  %645 = load float* %r119, align 4               ; <float> [#uses=1]
  %646 = fptoui float %645 to i8                  ; <i8> [#uses=1]
  store i8 %646, i8* %iftmp.170, align 1
  br label %bb136

bb136:                                            ; preds = %bb135, %bb134
  %647 = load i8* %iftmp.170, align 1             ; <i8> [#uses=1]
  store i8 %647, i8* %iftmp.169, align 1
  br label %bb138

bb137:                                            ; preds = %bb132
  store i8 0, i8* %iftmp.169, align 1
  br label %bb138

bb138:                                            ; preds = %bb137, %bb136
  %648 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %649 = getelementptr inbounds i8* %648, i32 1   ; <i8*> [#uses=1]
  %650 = load i8* %iftmp.169, align 1             ; <i8> [#uses=1]
  store i8 %650, i8* %649, align 1
  br label %bb139

bb139:                                            ; preds = %bb138, %bb131
  %651 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %652 = getelementptr inbounds %struct.GLDState* %651, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %653 = getelementptr inbounds %struct.GLDMaskMode* %652, i32 0, i32 3 ; <i8*> [#uses=1]
  %654 = load i8* %653, align 1                   ; <i8> [#uses=1]
  %655 = zext i8 %654 to i32                      ; <i32> [#uses=1]
  %656 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %657 = and i32 %655, %656                       ; <i32> [#uses=1]
  %658 = icmp ne i32 %657, 0                      ; <i1> [#uses=1]
  br i1 %658, label %bb140, label %bb147

bb140:                                            ; preds = %bb139
  %659 = load float* %g120, align 4               ; <float> [#uses=1]
  %660 = fcmp uge float %659, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %660, label %bb141, label %bb145

bb141:                                            ; preds = %bb140
  %661 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %662 = getelementptr inbounds %struct.GLDContextRec* %661, i32 0, i32 7 ; <float*> [#uses=1]
  %663 = load float* %662, align 4                ; <float> [#uses=1]
  %664 = load float* %g120, align 4               ; <float> [#uses=1]
  %665 = fcmp olt float %663, %664                ; <i1> [#uses=1]
  br i1 %665, label %bb142, label %bb143

bb142:                                            ; preds = %bb141
  %666 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %667 = getelementptr inbounds %struct.GLDContextRec* %666, i32 0, i32 7 ; <float*> [#uses=1]
  %668 = load float* %667, align 4                ; <float> [#uses=1]
  %669 = fptoui float %668 to i8                  ; <i8> [#uses=1]
  store i8 %669, i8* %iftmp.172, align 1
  br label %bb144

bb143:                                            ; preds = %bb141
  %670 = load float* %g120, align 4               ; <float> [#uses=1]
  %671 = fptoui float %670 to i8                  ; <i8> [#uses=1]
  store i8 %671, i8* %iftmp.172, align 1
  br label %bb144

bb144:                                            ; preds = %bb143, %bb142
  %672 = load i8* %iftmp.172, align 1             ; <i8> [#uses=1]
  store i8 %672, i8* %iftmp.171, align 1
  br label %bb146

bb145:                                            ; preds = %bb140
  store i8 0, i8* %iftmp.171, align 1
  br label %bb146

bb146:                                            ; preds = %bb145, %bb144
  %673 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %674 = getelementptr inbounds i8* %673, i32 2   ; <i8*> [#uses=1]
  %675 = load i8* %iftmp.171, align 1             ; <i8> [#uses=1]
  store i8 %675, i8* %674, align 1
  br label %bb147

bb147:                                            ; preds = %bb146, %bb139
  %676 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %677 = getelementptr inbounds %struct.GLDState* %676, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %678 = getelementptr inbounds %struct.GLDMaskMode* %677, i32 0, i32 4 ; <i8*> [#uses=1]
  %679 = load i8* %678, align 2                   ; <i8> [#uses=1]
  %680 = zext i8 %679 to i32                      ; <i32> [#uses=1]
  %681 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %682 = and i32 %680, %681                       ; <i32> [#uses=1]
  %683 = icmp ne i32 %682, 0                      ; <i1> [#uses=1]
  br i1 %683, label %bb148, label %bb155

bb148:                                            ; preds = %bb147
  %684 = load float* %b121, align 4               ; <float> [#uses=1]
  %685 = fcmp uge float %684, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %685, label %bb149, label %bb153

bb149:                                            ; preds = %bb148
  %686 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %687 = getelementptr inbounds %struct.GLDContextRec* %686, i32 0, i32 7 ; <float*> [#uses=1]
  %688 = load float* %687, align 4                ; <float> [#uses=1]
  %689 = load float* %b121, align 4               ; <float> [#uses=1]
  %690 = fcmp olt float %688, %689                ; <i1> [#uses=1]
  br i1 %690, label %bb150, label %bb151

bb150:                                            ; preds = %bb149
  %691 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %692 = getelementptr inbounds %struct.GLDContextRec* %691, i32 0, i32 7 ; <float*> [#uses=1]
  %693 = load float* %692, align 4                ; <float> [#uses=1]
  %694 = fptoui float %693 to i8                  ; <i8> [#uses=1]
  store i8 %694, i8* %iftmp.174, align 1
  br label %bb152

bb151:                                            ; preds = %bb149
  %695 = load float* %b121, align 4               ; <float> [#uses=1]
  %696 = fptoui float %695 to i8                  ; <i8> [#uses=1]
  store i8 %696, i8* %iftmp.174, align 1
  br label %bb152

bb152:                                            ; preds = %bb151, %bb150
  %697 = load i8* %iftmp.174, align 1             ; <i8> [#uses=1]
  store i8 %697, i8* %iftmp.173, align 1
  br label %bb154

bb153:                                            ; preds = %bb148
  store i8 0, i8* %iftmp.173, align 1
  br label %bb154

bb154:                                            ; preds = %bb153, %bb152
  %698 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %699 = getelementptr inbounds i8* %698, i32 3   ; <i8*> [#uses=1]
  %700 = load i8* %iftmp.173, align 1             ; <i8> [#uses=1]
  store i8 %700, i8* %699, align 1
  br label %bb155

bb155:                                            ; preds = %bb154, %bb147
  br label %bb188

bb156:                                            ; preds = %bb118
  %701 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %702 = getelementptr inbounds %struct.GLDState* %701, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %703 = getelementptr inbounds %struct.GLDMaskMode* %702, i32 0, i32 5 ; <i8*> [#uses=1]
  %704 = load i8* %703, align 1                   ; <i8> [#uses=1]
  %705 = zext i8 %704 to i32                      ; <i32> [#uses=1]
  %706 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %707 = and i32 %705, %706                       ; <i32> [#uses=1]
  %708 = icmp ne i32 %707, 0                      ; <i1> [#uses=1]
  br i1 %708, label %bb157, label %bb164

bb157:                                            ; preds = %bb156
  %709 = load float* %a122, align 4               ; <float> [#uses=1]
  %710 = fcmp uge float %709, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %710, label %bb158, label %bb162

bb158:                                            ; preds = %bb157
  %711 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %712 = getelementptr inbounds %struct.GLDContextRec* %711, i32 0, i32 7 ; <float*> [#uses=1]
  %713 = load float* %712, align 4                ; <float> [#uses=1]
  %714 = load float* %a122, align 4               ; <float> [#uses=1]
  %715 = fcmp olt float %713, %714                ; <i1> [#uses=1]
  br i1 %715, label %bb159, label %bb160

bb159:                                            ; preds = %bb158
  %716 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %717 = getelementptr inbounds %struct.GLDContextRec* %716, i32 0, i32 7 ; <float*> [#uses=1]
  %718 = load float* %717, align 4                ; <float> [#uses=1]
  %719 = fptoui float %718 to i8                  ; <i8> [#uses=1]
  store i8 %719, i8* %iftmp.176, align 1
  br label %bb161

bb160:                                            ; preds = %bb158
  %720 = load float* %a122, align 4               ; <float> [#uses=1]
  %721 = fptoui float %720 to i8                  ; <i8> [#uses=1]
  store i8 %721, i8* %iftmp.176, align 1
  br label %bb161

bb161:                                            ; preds = %bb160, %bb159
  %722 = load i8* %iftmp.176, align 1             ; <i8> [#uses=1]
  store i8 %722, i8* %iftmp.175, align 1
  br label %bb163

bb162:                                            ; preds = %bb157
  store i8 0, i8* %iftmp.175, align 1
  br label %bb163

bb163:                                            ; preds = %bb162, %bb161
  %723 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %724 = getelementptr inbounds i8* %723, i32 3   ; <i8*> [#uses=1]
  %725 = load i8* %iftmp.175, align 1             ; <i8> [#uses=1]
  store i8 %725, i8* %724, align 1
  br label %bb164

bb164:                                            ; preds = %bb163, %bb156
  %726 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %727 = getelementptr inbounds %struct.GLDState* %726, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %728 = getelementptr inbounds %struct.GLDMaskMode* %727, i32 0, i32 2 ; <i8*> [#uses=1]
  %729 = load i8* %728, align 16                  ; <i8> [#uses=1]
  %730 = zext i8 %729 to i32                      ; <i32> [#uses=1]
  %731 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %732 = and i32 %730, %731                       ; <i32> [#uses=1]
  %733 = icmp ne i32 %732, 0                      ; <i1> [#uses=1]
  br i1 %733, label %bb165, label %bb172

bb165:                                            ; preds = %bb164
  %734 = load float* %r119, align 4               ; <float> [#uses=1]
  %735 = fcmp uge float %734, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %735, label %bb166, label %bb170

bb166:                                            ; preds = %bb165
  %736 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %737 = getelementptr inbounds %struct.GLDContextRec* %736, i32 0, i32 7 ; <float*> [#uses=1]
  %738 = load float* %737, align 4                ; <float> [#uses=1]
  %739 = load float* %r119, align 4               ; <float> [#uses=1]
  %740 = fcmp olt float %738, %739                ; <i1> [#uses=1]
  br i1 %740, label %bb167, label %bb168

bb167:                                            ; preds = %bb166
  %741 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %742 = getelementptr inbounds %struct.GLDContextRec* %741, i32 0, i32 7 ; <float*> [#uses=1]
  %743 = load float* %742, align 4                ; <float> [#uses=1]
  %744 = fptoui float %743 to i8                  ; <i8> [#uses=1]
  store i8 %744, i8* %iftmp.178, align 1
  br label %bb169

bb168:                                            ; preds = %bb166
  %745 = load float* %r119, align 4               ; <float> [#uses=1]
  %746 = fptoui float %745 to i8                  ; <i8> [#uses=1]
  store i8 %746, i8* %iftmp.178, align 1
  br label %bb169

bb169:                                            ; preds = %bb168, %bb167
  %747 = load i8* %iftmp.178, align 1             ; <i8> [#uses=1]
  store i8 %747, i8* %iftmp.177, align 1
  br label %bb171

bb170:                                            ; preds = %bb165
  store i8 0, i8* %iftmp.177, align 1
  br label %bb171

bb171:                                            ; preds = %bb170, %bb169
  %748 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %749 = getelementptr inbounds i8* %748, i32 2   ; <i8*> [#uses=1]
  %750 = load i8* %iftmp.177, align 1             ; <i8> [#uses=1]
  store i8 %750, i8* %749, align 1
  br label %bb172

bb172:                                            ; preds = %bb171, %bb164
  %751 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %752 = getelementptr inbounds %struct.GLDState* %751, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %753 = getelementptr inbounds %struct.GLDMaskMode* %752, i32 0, i32 3 ; <i8*> [#uses=1]
  %754 = load i8* %753, align 1                   ; <i8> [#uses=1]
  %755 = zext i8 %754 to i32                      ; <i32> [#uses=1]
  %756 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %757 = and i32 %755, %756                       ; <i32> [#uses=1]
  %758 = icmp ne i32 %757, 0                      ; <i1> [#uses=1]
  br i1 %758, label %bb173, label %bb180

bb173:                                            ; preds = %bb172
  %759 = load float* %g120, align 4               ; <float> [#uses=1]
  %760 = fcmp uge float %759, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %760, label %bb174, label %bb178

bb174:                                            ; preds = %bb173
  %761 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %762 = getelementptr inbounds %struct.GLDContextRec* %761, i32 0, i32 7 ; <float*> [#uses=1]
  %763 = load float* %762, align 4                ; <float> [#uses=1]
  %764 = load float* %g120, align 4               ; <float> [#uses=1]
  %765 = fcmp olt float %763, %764                ; <i1> [#uses=1]
  br i1 %765, label %bb175, label %bb176

bb175:                                            ; preds = %bb174
  %766 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %767 = getelementptr inbounds %struct.GLDContextRec* %766, i32 0, i32 7 ; <float*> [#uses=1]
  %768 = load float* %767, align 4                ; <float> [#uses=1]
  %769 = fptoui float %768 to i8                  ; <i8> [#uses=1]
  store i8 %769, i8* %iftmp.180, align 1
  br label %bb177

bb176:                                            ; preds = %bb174
  %770 = load float* %g120, align 4               ; <float> [#uses=1]
  %771 = fptoui float %770 to i8                  ; <i8> [#uses=1]
  store i8 %771, i8* %iftmp.180, align 1
  br label %bb177

bb177:                                            ; preds = %bb176, %bb175
  %772 = load i8* %iftmp.180, align 1             ; <i8> [#uses=1]
  store i8 %772, i8* %iftmp.179, align 1
  br label %bb179

bb178:                                            ; preds = %bb173
  store i8 0, i8* %iftmp.179, align 1
  br label %bb179

bb179:                                            ; preds = %bb178, %bb177
  %773 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %774 = getelementptr inbounds i8* %773, i32 1   ; <i8*> [#uses=1]
  %775 = load i8* %iftmp.179, align 1             ; <i8> [#uses=1]
  store i8 %775, i8* %774, align 1
  br label %bb180

bb180:                                            ; preds = %bb179, %bb172
  %776 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %777 = getelementptr inbounds %struct.GLDState* %776, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %778 = getelementptr inbounds %struct.GLDMaskMode* %777, i32 0, i32 4 ; <i8*> [#uses=1]
  %779 = load i8* %778, align 2                   ; <i8> [#uses=1]
  %780 = zext i8 %779 to i32                      ; <i32> [#uses=1]
  %781 = load i32* %cur_draw_buffer_mask_bit115, align 4 ; <i32> [#uses=1]
  %782 = and i32 %780, %781                       ; <i32> [#uses=1]
  %783 = icmp ne i32 %782, 0                      ; <i1> [#uses=1]
  br i1 %783, label %bb181, label %bb188

bb181:                                            ; preds = %bb180
  %784 = load float* %b121, align 4               ; <float> [#uses=1]
  %785 = fcmp uge float %784, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %785, label %bb182, label %bb186

bb182:                                            ; preds = %bb181
  %786 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %787 = getelementptr inbounds %struct.GLDContextRec* %786, i32 0, i32 7 ; <float*> [#uses=1]
  %788 = load float* %787, align 4                ; <float> [#uses=1]
  %789 = load float* %b121, align 4               ; <float> [#uses=1]
  %790 = fcmp olt float %788, %789                ; <i1> [#uses=1]
  br i1 %790, label %bb183, label %bb184

bb183:                                            ; preds = %bb182
  %791 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %792 = getelementptr inbounds %struct.GLDContextRec* %791, i32 0, i32 7 ; <float*> [#uses=1]
  %793 = load float* %792, align 4                ; <float> [#uses=1]
  %794 = fptoui float %793 to i8                  ; <i8> [#uses=1]
  store i8 %794, i8* %iftmp.182, align 1
  br label %bb185

bb184:                                            ; preds = %bb182
  %795 = load float* %b121, align 4               ; <float> [#uses=1]
  %796 = fptoui float %795 to i8                  ; <i8> [#uses=1]
  store i8 %796, i8* %iftmp.182, align 1
  br label %bb185

bb185:                                            ; preds = %bb184, %bb183
  %797 = load i8* %iftmp.182, align 1             ; <i8> [#uses=1]
  store i8 %797, i8* %iftmp.181, align 1
  br label %bb187

bb186:                                            ; preds = %bb181
  store i8 0, i8* %iftmp.181, align 1
  br label %bb187

bb187:                                            ; preds = %bb186, %bb185
  %798 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %799 = getelementptr inbounds i8* %798, i32 0   ; <i8*> [#uses=1]
  %800 = load i8* %iftmp.181, align 1             ; <i8> [#uses=1]
  store i8 %800, i8* %799, align 1
  br label %bb188

bb188:                                            ; preds = %bb187, %bb180, %bb155
  %801 = load double** %accum, align 4            ; <double*> [#uses=1]
  %802 = getelementptr inbounds double* %801, i32 4 ; <double*> [#uses=1]
  store double* %802, double** %accum, align 4
  %803 = load i8** %color_ptr111, align 4         ; <i8*> [#uses=1]
  %804 = getelementptr inbounds i8* %803, i32 4   ; <i8*> [#uses=1]
  store i8* %804, i8** %color_ptr111, align 4
  br label %bb189

bb189:                                            ; preds = %bb188, %bb117
  %805 = load double** %accum, align 4            ; <double*> [#uses=1]
  %806 = load double** %accum_end, align 4        ; <double*> [#uses=1]
  %807 = icmp ult double* %805, %806              ; <i1> [#uses=1]
  br i1 %807, label %bb118, label %bb190

bb190:                                            ; preds = %bb189
  br label %bb224

bb191:                                            ; preds = %bb116
  br label %bb223

bb192:                                            ; preds = %bb223
  %808 = load double** %accum, align 4            ; <double*> [#uses=1]
  %809 = getelementptr inbounds double* %808, i32 0 ; <double*> [#uses=1]
  %810 = load double* %809, align 1               ; <double> [#uses=1]
  %811 = fptrunc double %810 to float             ; <float> [#uses=1]
  %812 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %813 = fmul float %811, %812                    ; <float> [#uses=1]
  %814 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %815 = fmul float %813, %814                    ; <float> [#uses=1]
  %816 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %817 = getelementptr inbounds %struct.GLDContextRec* %816, i32 0, i32 1 ; <float*> [#uses=1]
  %818 = load float* %817, align 4                ; <float> [#uses=1]
  %819 = fadd float %815, %818                    ; <float> [#uses=1]
  %820 = fcmp uge float %819, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %820, label %bb197, label %bb201

bb197:                                            ; preds = %bb192
  %821 = load double** %accum, align 4            ; <double*> [#uses=1]
  %822 = getelementptr inbounds double* %821, i32 0 ; <double*> [#uses=1]
  %823 = load double* %822, align 1               ; <double> [#uses=1]
  %824 = fptrunc double %823 to float             ; <float> [#uses=1]
  %825 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %826 = fmul float %824, %825                    ; <float> [#uses=1]
  %827 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %828 = fmul float %826, %827                    ; <float> [#uses=1]
  %829 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %830 = getelementptr inbounds %struct.GLDContextRec* %829, i32 0, i32 1 ; <float*> [#uses=1]
  %831 = load float* %830, align 4                ; <float> [#uses=1]
  %832 = fadd float %828, %831                    ; <float> [#uses=1]
  %833 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %834 = getelementptr inbounds %struct.GLDContextRec* %833, i32 0, i32 7 ; <float*> [#uses=1]
  %835 = load float* %834, align 4                ; <float> [#uses=1]
  %836 = fcmp ogt float %832, %835                ; <i1> [#uses=1]
  br i1 %836, label %bb198, label %bb199

bb198:                                            ; preds = %bb197
  %837 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %838 = getelementptr inbounds %struct.GLDContextRec* %837, i32 0, i32 7 ; <float*> [#uses=1]
  %839 = load float* %838, align 4                ; <float> [#uses=1]
  %840 = fptoui float %839 to i8                  ; <i8> [#uses=1]
  %841 = zext i8 %840 to i32                      ; <i32> [#uses=1]
  store i32 %841, i32* %iftmp.184, align 4
  br label %bb200

bb199:                                            ; preds = %bb197
  %842 = load double** %accum, align 4            ; <double*> [#uses=1]
  %843 = getelementptr inbounds double* %842, i32 0 ; <double*> [#uses=1]
  %844 = load double* %843, align 1               ; <double> [#uses=1]
  %845 = fptrunc double %844 to float             ; <float> [#uses=1]
  %846 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %847 = fmul float %845, %846                    ; <float> [#uses=1]
  %848 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %849 = fmul float %847, %848                    ; <float> [#uses=1]
  %850 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %851 = getelementptr inbounds %struct.GLDContextRec* %850, i32 0, i32 1 ; <float*> [#uses=1]
  %852 = load float* %851, align 4                ; <float> [#uses=1]
  %853 = fadd float %849, %852                    ; <float> [#uses=1]
  %854 = fptoui float %853 to i8                  ; <i8> [#uses=1]
  %855 = zext i8 %854 to i32                      ; <i32> [#uses=1]
  store i32 %855, i32* %iftmp.184, align 4
  br label %bb200

bb200:                                            ; preds = %bb199, %bb198
  %856 = load i32* %iftmp.184, align 4            ; <i32> [#uses=1]
  store i32 %856, i32* %iftmp.183, align 4
  br label %bb202

bb201:                                            ; preds = %bb192
  store i32 0, i32* %iftmp.183, align 4
  br label %bb202

bb202:                                            ; preds = %bb201, %bb200
  %857 = load i32* %iftmp.183, align 4            ; <i32> [#uses=1]
  store i32 %857, i32* %r193, align 4
  %858 = load double** %accum, align 4            ; <double*> [#uses=1]
  %859 = getelementptr inbounds double* %858, i32 1 ; <double*> [#uses=1]
  %860 = load double* %859, align 1               ; <double> [#uses=1]
  %861 = fptrunc double %860 to float             ; <float> [#uses=1]
  %862 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %863 = fmul float %861, %862                    ; <float> [#uses=1]
  %864 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %865 = fmul float %863, %864                    ; <float> [#uses=1]
  %866 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %867 = getelementptr inbounds %struct.GLDContextRec* %866, i32 0, i32 1 ; <float*> [#uses=1]
  %868 = load float* %867, align 4                ; <float> [#uses=1]
  %869 = fadd float %865, %868                    ; <float> [#uses=1]
  %870 = fcmp uge float %869, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %870, label %bb203, label %bb207

bb203:                                            ; preds = %bb202
  %871 = load double** %accum, align 4            ; <double*> [#uses=1]
  %872 = getelementptr inbounds double* %871, i32 1 ; <double*> [#uses=1]
  %873 = load double* %872, align 1               ; <double> [#uses=1]
  %874 = fptrunc double %873 to float             ; <float> [#uses=1]
  %875 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %876 = fmul float %874, %875                    ; <float> [#uses=1]
  %877 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %878 = fmul float %876, %877                    ; <float> [#uses=1]
  %879 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %880 = getelementptr inbounds %struct.GLDContextRec* %879, i32 0, i32 1 ; <float*> [#uses=1]
  %881 = load float* %880, align 4                ; <float> [#uses=1]
  %882 = fadd float %878, %881                    ; <float> [#uses=1]
  %883 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %884 = getelementptr inbounds %struct.GLDContextRec* %883, i32 0, i32 7 ; <float*> [#uses=1]
  %885 = load float* %884, align 4                ; <float> [#uses=1]
  %886 = fcmp ogt float %882, %885                ; <i1> [#uses=1]
  br i1 %886, label %bb204, label %bb205

bb204:                                            ; preds = %bb203
  %887 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %888 = getelementptr inbounds %struct.GLDContextRec* %887, i32 0, i32 7 ; <float*> [#uses=1]
  %889 = load float* %888, align 4                ; <float> [#uses=1]
  %890 = fptoui float %889 to i8                  ; <i8> [#uses=1]
  %891 = zext i8 %890 to i32                      ; <i32> [#uses=1]
  store i32 %891, i32* %iftmp.186, align 4
  br label %bb206

bb205:                                            ; preds = %bb203
  %892 = load double** %accum, align 4            ; <double*> [#uses=1]
  %893 = getelementptr inbounds double* %892, i32 1 ; <double*> [#uses=1]
  %894 = load double* %893, align 1               ; <double> [#uses=1]
  %895 = fptrunc double %894 to float             ; <float> [#uses=1]
  %896 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %897 = fmul float %895, %896                    ; <float> [#uses=1]
  %898 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %899 = fmul float %897, %898                    ; <float> [#uses=1]
  %900 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %901 = getelementptr inbounds %struct.GLDContextRec* %900, i32 0, i32 1 ; <float*> [#uses=1]
  %902 = load float* %901, align 4                ; <float> [#uses=1]
  %903 = fadd float %899, %902                    ; <float> [#uses=1]
  %904 = fptoui float %903 to i8                  ; <i8> [#uses=1]
  %905 = zext i8 %904 to i32                      ; <i32> [#uses=1]
  store i32 %905, i32* %iftmp.186, align 4
  br label %bb206

bb206:                                            ; preds = %bb205, %bb204
  %906 = load i32* %iftmp.186, align 4            ; <i32> [#uses=1]
  store i32 %906, i32* %iftmp.185, align 4
  br label %bb208

bb207:                                            ; preds = %bb202
  store i32 0, i32* %iftmp.185, align 4
  br label %bb208

bb208:                                            ; preds = %bb207, %bb206
  %907 = load i32* %iftmp.185, align 4            ; <i32> [#uses=1]
  store i32 %907, i32* %g194, align 4
  %908 = load double** %accum, align 4            ; <double*> [#uses=1]
  %909 = getelementptr inbounds double* %908, i32 2 ; <double*> [#uses=1]
  %910 = load double* %909, align 1               ; <double> [#uses=1]
  %911 = fptrunc double %910 to float             ; <float> [#uses=1]
  %912 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %913 = fmul float %911, %912                    ; <float> [#uses=1]
  %914 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %915 = fmul float %913, %914                    ; <float> [#uses=1]
  %916 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %917 = getelementptr inbounds %struct.GLDContextRec* %916, i32 0, i32 1 ; <float*> [#uses=1]
  %918 = load float* %917, align 4                ; <float> [#uses=1]
  %919 = fadd float %915, %918                    ; <float> [#uses=1]
  %920 = fcmp uge float %919, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %920, label %bb209, label %bb213

bb209:                                            ; preds = %bb208
  %921 = load double** %accum, align 4            ; <double*> [#uses=1]
  %922 = getelementptr inbounds double* %921, i32 2 ; <double*> [#uses=1]
  %923 = load double* %922, align 1               ; <double> [#uses=1]
  %924 = fptrunc double %923 to float             ; <float> [#uses=1]
  %925 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %926 = fmul float %924, %925                    ; <float> [#uses=1]
  %927 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %928 = fmul float %926, %927                    ; <float> [#uses=1]
  %929 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %930 = getelementptr inbounds %struct.GLDContextRec* %929, i32 0, i32 1 ; <float*> [#uses=1]
  %931 = load float* %930, align 4                ; <float> [#uses=1]
  %932 = fadd float %928, %931                    ; <float> [#uses=1]
  %933 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %934 = getelementptr inbounds %struct.GLDContextRec* %933, i32 0, i32 7 ; <float*> [#uses=1]
  %935 = load float* %934, align 4                ; <float> [#uses=1]
  %936 = fcmp ogt float %932, %935                ; <i1> [#uses=1]
  br i1 %936, label %bb210, label %bb211

bb210:                                            ; preds = %bb209
  %937 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %938 = getelementptr inbounds %struct.GLDContextRec* %937, i32 0, i32 7 ; <float*> [#uses=1]
  %939 = load float* %938, align 4                ; <float> [#uses=1]
  %940 = fptoui float %939 to i8                  ; <i8> [#uses=1]
  %941 = zext i8 %940 to i32                      ; <i32> [#uses=1]
  store i32 %941, i32* %iftmp.188, align 4
  br label %bb212

bb211:                                            ; preds = %bb209
  %942 = load double** %accum, align 4            ; <double*> [#uses=1]
  %943 = getelementptr inbounds double* %942, i32 2 ; <double*> [#uses=1]
  %944 = load double* %943, align 1               ; <double> [#uses=1]
  %945 = fptrunc double %944 to float             ; <float> [#uses=1]
  %946 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %947 = fmul float %945, %946                    ; <float> [#uses=1]
  %948 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %949 = fmul float %947, %948                    ; <float> [#uses=1]
  %950 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %951 = getelementptr inbounds %struct.GLDContextRec* %950, i32 0, i32 1 ; <float*> [#uses=1]
  %952 = load float* %951, align 4                ; <float> [#uses=1]
  %953 = fadd float %949, %952                    ; <float> [#uses=1]
  %954 = fptoui float %953 to i8                  ; <i8> [#uses=1]
  %955 = zext i8 %954 to i32                      ; <i32> [#uses=1]
  store i32 %955, i32* %iftmp.188, align 4
  br label %bb212

bb212:                                            ; preds = %bb211, %bb210
  %956 = load i32* %iftmp.188, align 4            ; <i32> [#uses=1]
  store i32 %956, i32* %iftmp.187, align 4
  br label %bb214

bb213:                                            ; preds = %bb208
  store i32 0, i32* %iftmp.187, align 4
  br label %bb214

bb214:                                            ; preds = %bb213, %bb212
  %957 = load i32* %iftmp.187, align 4            ; <i32> [#uses=1]
  store i32 %957, i32* %b195, align 4
  %958 = load double** %accum, align 4            ; <double*> [#uses=1]
  %959 = getelementptr inbounds double* %958, i32 3 ; <double*> [#uses=1]
  %960 = load double* %959, align 1               ; <double> [#uses=1]
  %961 = fptrunc double %960 to float             ; <float> [#uses=1]
  %962 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %963 = fmul float %961, %962                    ; <float> [#uses=1]
  %964 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %965 = fmul float %963, %964                    ; <float> [#uses=1]
  %966 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %967 = getelementptr inbounds %struct.GLDContextRec* %966, i32 0, i32 1 ; <float*> [#uses=1]
  %968 = load float* %967, align 4                ; <float> [#uses=1]
  %969 = fadd float %965, %968                    ; <float> [#uses=1]
  %970 = fcmp uge float %969, 0.000000e+00        ; <i1> [#uses=1]
  br i1 %970, label %bb215, label %bb219

bb215:                                            ; preds = %bb214
  %971 = load double** %accum, align 4            ; <double*> [#uses=1]
  %972 = getelementptr inbounds double* %971, i32 3 ; <double*> [#uses=1]
  %973 = load double* %972, align 1               ; <double> [#uses=1]
  %974 = fptrunc double %973 to float             ; <float> [#uses=1]
  %975 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %976 = fmul float %974, %975                    ; <float> [#uses=1]
  %977 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %978 = fmul float %976, %977                    ; <float> [#uses=1]
  %979 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %980 = getelementptr inbounds %struct.GLDContextRec* %979, i32 0, i32 1 ; <float*> [#uses=1]
  %981 = load float* %980, align 4                ; <float> [#uses=1]
  %982 = fadd float %978, %981                    ; <float> [#uses=1]
  %983 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %984 = getelementptr inbounds %struct.GLDContextRec* %983, i32 0, i32 7 ; <float*> [#uses=1]
  %985 = load float* %984, align 4                ; <float> [#uses=1]
  %986 = fcmp ogt float %982, %985                ; <i1> [#uses=1]
  br i1 %986, label %bb216, label %bb217

bb216:                                            ; preds = %bb215
  %987 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %988 = getelementptr inbounds %struct.GLDContextRec* %987, i32 0, i32 7 ; <float*> [#uses=1]
  %989 = load float* %988, align 4                ; <float> [#uses=1]
  %990 = fptoui float %989 to i8                  ; <i8> [#uses=1]
  %991 = zext i8 %990 to i32                      ; <i32> [#uses=1]
  store i32 %991, i32* %iftmp.190, align 4
  br label %bb218

bb217:                                            ; preds = %bb215
  %992 = load double** %accum, align 4            ; <double*> [#uses=1]
  %993 = getelementptr inbounds double* %992, i32 3 ; <double*> [#uses=1]
  %994 = load double* %993, align 1               ; <double> [#uses=1]
  %995 = fptrunc double %994 to float             ; <float> [#uses=1]
  %996 = load float* %value_addr, align 4         ; <float> [#uses=1]
  %997 = fmul float %995, %996                    ; <float> [#uses=1]
  %998 = load float* %twoFiftyFive, align 4       ; <float> [#uses=1]
  %999 = fmul float %997, %998                    ; <float> [#uses=1]
  %1000 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1001 = getelementptr inbounds %struct.GLDContextRec* %1000, i32 0, i32 1 ; <float*> [#uses=1]
  %1002 = load float* %1001, align 4              ; <float> [#uses=1]
  %1003 = fadd float %999, %1002                  ; <float> [#uses=1]
  %1004 = fptoui float %1003 to i8                ; <i8> [#uses=1]
  %1005 = zext i8 %1004 to i32                    ; <i32> [#uses=1]
  store i32 %1005, i32* %iftmp.190, align 4
  br label %bb218

bb218:                                            ; preds = %bb217, %bb216
  %1006 = load i32* %iftmp.190, align 4           ; <i32> [#uses=1]
  store i32 %1006, i32* %iftmp.189, align 4
  br label %bb220

bb219:                                            ; preds = %bb214
  store i32 0, i32* %iftmp.189, align 4
  br label %bb220

bb220:                                            ; preds = %bb219, %bb218
  %1007 = load i32* %iftmp.189, align 4           ; <i32> [#uses=1]
  store i32 %1007, i32* %a196, align 4
  %1008 = load i32* %a196, align 4                ; <i32> [#uses=1]
  %1009 = shl i32 %1008, 24                       ; <i32> [#uses=1]
  %1010 = load i32* %r193, align 4                ; <i32> [#uses=1]
  %1011 = shl i32 %1010, 16                       ; <i32> [#uses=1]
  %1012 = or i32 %1009, %1011                     ; <i32> [#uses=1]
  %1013 = load i32* %g194, align 4                ; <i32> [#uses=1]
  %1014 = shl i32 %1013, 8                        ; <i32> [#uses=1]
  %1015 = or i32 %1012, %1014                     ; <i32> [#uses=1]
  %1016 = load i32* %b195, align 4                ; <i32> [#uses=1]
  %1017 = or i32 %1015, %1016                     ; <i32> [#uses=1]
  store i32 %1017, i32* %color, align 4
  %1018 = load i8* %swap, align 1                 ; <i8> [#uses=1]
  %1019 = icmp ne i8 %1018, 0                     ; <i1> [#uses=1]
  br i1 %1019, label %bb221, label %bb222

bb221:                                            ; preds = %bb220
  %1020 = load i32* %color, align 4               ; <i32> [#uses=1]
  %1021 = call i32 @_OSSwapInt32(i32 %1020) nounwind inlinehint ssp ; <i32> [#uses=1]
  store i32 %1021, i32* %color, align 4
  br label %bb222

bb222:                                            ; preds = %bb221, %bb220
  %1022 = load i8** %color_ptr111, align 4        ; <i8*> [#uses=1]
  %1023 = bitcast i8* %1022 to i32*               ; <i32*> [#uses=1]
  %1024 = getelementptr inbounds i32* %1023, i32 0 ; <i32*> [#uses=1]
  %1025 = load i32* %color, align 4               ; <i32> [#uses=1]
  store i32 %1025, i32* %1024, align 1
  %1026 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1027 = getelementptr inbounds double* %1026, i32 4 ; <double*> [#uses=1]
  store double* %1027, double** %accum, align 4
  %1028 = load i8** %color_ptr111, align 4        ; <i8*> [#uses=1]
  %1029 = getelementptr inbounds i8* %1028, i32 4 ; <i8*> [#uses=1]
  store i8* %1029, i8** %color_ptr111, align 4
  br label %bb223

bb223:                                            ; preds = %bb222, %bb191
  %1030 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1031 = load double** %accum_end, align 4       ; <double*> [#uses=1]
  %1032 = icmp ult double* %1030, %1031           ; <i1> [#uses=1]
  br i1 %1032, label %bb192, label %bb224

bb224:                                            ; preds = %bb223, %bb190
  %1033 = load i32* %y, align 4                   ; <i32> [#uses=1]
  %1034 = add i32 %1033, 1                        ; <i32> [#uses=1]
  store i32 %1034, i32* %y, align 4
  %1035 = load i32* %offset, align 4              ; <i32> [#uses=1]
  %1036 = load i32* %y_inc, align 4               ; <i32> [#uses=1]
  %1037 = add i32 %1035, %1036                    ; <i32> [#uses=1]
  store i32 %1037, i32* %offset, align 4
  br label %bb225

bb225:                                            ; preds = %bb224, %bb114
  %1038 = load i32* %y, align 4                   ; <i32> [#uses=1]
  %1039 = load i32* %yl, align 4                  ; <i32> [#uses=1]
  %1040 = icmp ult i32 %1038, %1039               ; <i1> [#uses=1]
  br i1 %1040, label %bb116, label %bb226

bb226:                                            ; preds = %bb225, %bb113
  %1041 = load i32* %draw_buffer, align 4         ; <i32> [#uses=1]
  %1042 = add nsw i32 %1041, 1                    ; <i32> [#uses=1]
  store i32 %1042, i32* %draw_buffer, align 4
  br label %bb227

bb227:                                            ; preds = %bb226, %bb110
  %1043 = load i32* %draw_buffer, align 4         ; <i32> [#uses=1]
  %1044 = icmp sle i32 %1043, 7                   ; <i1> [#uses=1]
  br i1 %1044, label %bb113, label %bb228

bb228:                                            ; preds = %bb227
  br label %bb316

bb229:                                            ; preds = %bb108
  %1045 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1046 = getelementptr inbounds %struct.GLDContextRec* %1045, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %1047 = load %struct.GLDFramebufferRec** %1046, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %1048 = icmp ne %struct.GLDFramebufferRec* %1047, null ; <i1> [#uses=1]
  br i1 %1048, label %bb230, label %bb231

bb230:                                            ; preds = %bb229
  %1049 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1050 = getelementptr inbounds %struct.GLDContextRec* %1049, i32 0, i32 28 ; <%struct.GLDFramebufferRec**> [#uses=1]
  %1051 = load %struct.GLDFramebufferRec** %1050, align 4 ; <%struct.GLDFramebufferRec*> [#uses=1]
  %1052 = getelementptr inbounds %struct.GLDFramebufferRec* %1051, i32 0, i32 2 ; <[10 x %struct.GLDFormat]*> [#uses=1]
  %1053 = getelementptr inbounds [10 x %struct.GLDFormat]* %1052, i32 0, i32 0 ; <%struct.GLDFormat*> [#uses=1]
  %1054 = getelementptr inbounds %struct.GLDFormat* %1053, i32 0, i32 7 ; <i32*> [#uses=1]
  %1055 = load i32* %1054, align 4                ; <i32> [#uses=1]
  %1056 = icmp eq i32 %1055, 5123                 ; <i1> [#uses=1]
  %1057 = zext i1 %1056 to i8                     ; <i8> [#uses=1]
  store i8 %1057, i8* %iftmp.192, align 1
  br label %bb232

bb231:                                            ; preds = %bb229
  %1058 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1059 = getelementptr inbounds %struct.GLDContextRec* %1058, i32 0, i32 56 ; <%struct.GLDFormat*> [#uses=1]
  %1060 = getelementptr inbounds %struct.GLDFormat* %1059, i32 0, i32 7 ; <i32*> [#uses=1]
  %1061 = load i32* %1060, align 4                ; <i32> [#uses=1]
  %1062 = icmp eq i32 %1061, 5123                 ; <i1> [#uses=1]
  %1063 = zext i1 %1062 to i8                     ; <i8> [#uses=1]
  store i8 %1063, i8* %iftmp.192, align 1
  br label %bb232

bb232:                                            ; preds = %bb231, %bb230
  %1064 = load i8* %iftmp.192, align 1            ; <i8> [#uses=1]
  %toBool233 = icmp ne i8 %1064, 0                ; <i1> [#uses=1]
  br i1 %toBool233, label %bb316, label %bb234

bb234:                                            ; preds = %bb232
  %1065 = load i32* %offset, align 4              ; <i32> [#uses=1]
  store i32 %1065, i32* %start_offset236, align 4
  store i32 0, i32* %draw_buffer, align 4
  br label %bb315

bb237:                                            ; preds = %bb315
  %1066 = load i32* %draw_buffer, align 4         ; <i32> [#uses=1]
  %1067 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1068 = getelementptr inbounds %struct.GLDContextRec* %1067, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %1069 = getelementptr inbounds %struct.GLDBufferstate* %1068, i32 0, i32 12 ; <[8 x %union.GLSBuffer]*> [#uses=1]
  %1070 = getelementptr inbounds [8 x %union.GLSBuffer]* %1069, i32 0, i32 %1066 ; <%union.GLSBuffer*> [#uses=1]
  %1071 = getelementptr inbounds %union.GLSBuffer* %1070, i32 0, i32 0 ; <i8**> [#uses=1]
  %1072 = load i8** %1071, align 4                ; <i8*> [#uses=1]
  %1073 = icmp ne i8* %1072, null                 ; <i1> [#uses=1]
  br i1 %1073, label %bb238, label %bb314

bb238:                                            ; preds = %bb237
  %1074 = load i32* %draw_buffer, align 4         ; <i32> [#uses=1]
  %1075 = shl i32 1, %1074                        ; <i32> [#uses=1]
  store i32 %1075, i32* %cur_draw_buffer_mask_bit239, align 4
  %1076 = load i32* %start_offset236, align 4     ; <i32> [#uses=1]
  store i32 %1076, i32* %offset, align 4
  %1077 = load i32* %cy, align 4                  ; <i32> [#uses=1]
  store i32 %1077, i32* %y, align 4
  br label %bb313

bb240:                                            ; preds = %bb313
  %1078 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1079 = getelementptr inbounds %struct.GLDContextRec* %1078, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %1080 = getelementptr inbounds %struct.GLDBufferstate* %1079, i32 0, i32 11 ; <%union.GLSBuffer*> [#uses=1]
  %1081 = getelementptr inbounds %union.GLSBuffer* %1080, i32 0, i32 0 ; <i8**> [#uses=1]
  %1082 = bitcast i8** %1081 to double**          ; <double**> [#uses=1]
  %1083 = load double** %1082, align 4            ; <double*> [#uses=1]
  %1084 = load i32* %offset, align 4              ; <i32> [#uses=1]
  %1085 = mul i32 %1084, 4                        ; <i32> [#uses=1]
  %1086 = getelementptr inbounds double* %1083, i32 %1085 ; <double*> [#uses=1]
  store double* %1086, double** %accum, align 4
  %1087 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1088 = load i32* %cw4, align 4                 ; <i32> [#uses=1]
  %1089 = getelementptr inbounds double* %1087, i32 %1088 ; <double*> [#uses=1]
  store double* %1089, double** %accum_end, align 4
  %1090 = load i32* %draw_buffer, align 4         ; <i32> [#uses=1]
  %1091 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1092 = getelementptr inbounds %struct.GLDContextRec* %1091, i32 0, i32 30 ; <%struct.GLDBufferstate*> [#uses=1]
  %1093 = getelementptr inbounds %struct.GLDBufferstate* %1092, i32 0, i32 12 ; <[8 x %union.GLSBuffer]*> [#uses=1]
  %1094 = getelementptr inbounds [8 x %union.GLSBuffer]* %1093, i32 0, i32 %1090 ; <%union.GLSBuffer*> [#uses=1]
  %1095 = getelementptr inbounds %union.GLSBuffer* %1094, i32 0, i32 0 ; <i8**> [#uses=1]
  %1096 = bitcast i8** %1095 to float**           ; <float**> [#uses=1]
  %1097 = load float** %1096, align 4             ; <float*> [#uses=1]
  %1098 = load i32* %offset, align 4              ; <i32> [#uses=1]
  %1099 = mul i32 %1098, 4                        ; <i32> [#uses=1]
  %1100 = getelementptr inbounds float* %1097, i32 %1099 ; <float*> [#uses=1]
  store float* %1100, float** %color_ptr235, align 4
  %1101 = load i8* %color_mask_enabled, align 1   ; <i8> [#uses=1]
  %1102 = icmp ne i8 %1101, 0                     ; <i1> [#uses=1]
  br i1 %1102, label %bb241, label %bb281

bb241:                                            ; preds = %bb240
  br label %bb279

bb242:                                            ; preds = %bb279
  %1103 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1104 = getelementptr inbounds double* %1103, i32 0 ; <double*> [#uses=1]
  %1105 = load double* %1104, align 1             ; <double> [#uses=1]
  %1106 = fptrunc double %1105 to float           ; <float> [#uses=1]
  %1107 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1108 = fmul float %1106, %1107                 ; <float> [#uses=1]
  store float %1108, float* %r243, align 4
  %1109 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1110 = getelementptr inbounds double* %1109, i32 1 ; <double*> [#uses=1]
  %1111 = load double* %1110, align 1             ; <double> [#uses=1]
  %1112 = fptrunc double %1111 to float           ; <float> [#uses=1]
  %1113 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1114 = fmul float %1112, %1113                 ; <float> [#uses=1]
  store float %1114, float* %g244, align 4
  %1115 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1116 = getelementptr inbounds double* %1115, i32 2 ; <double*> [#uses=1]
  %1117 = load double* %1116, align 1             ; <double> [#uses=1]
  %1118 = fptrunc double %1117 to float           ; <float> [#uses=1]
  %1119 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1120 = fmul float %1118, %1119                 ; <float> [#uses=1]
  store float %1120, float* %b245, align 4
  %1121 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1122 = getelementptr inbounds double* %1121, i32 3 ; <double*> [#uses=1]
  %1123 = load double* %1122, align 1             ; <double> [#uses=1]
  %1124 = fptrunc double %1123 to float           ; <float> [#uses=1]
  %1125 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1126 = fmul float %1124, %1125                 ; <float> [#uses=1]
  store float %1126, float* %a246, align 4
  %1127 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %1128 = getelementptr inbounds %struct.GLDState* %1127, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %1129 = getelementptr inbounds %struct.GLDMaskMode* %1128, i32 0, i32 2 ; <i8*> [#uses=1]
  %1130 = load i8* %1129, align 16                ; <i8> [#uses=1]
  %1131 = zext i8 %1130 to i32                    ; <i32> [#uses=1]
  %1132 = load i32* %cur_draw_buffer_mask_bit239, align 4 ; <i32> [#uses=1]
  %1133 = and i32 %1131, %1132                    ; <i32> [#uses=1]
  %1134 = icmp ne i32 %1133, 0                    ; <i1> [#uses=1]
  br i1 %1134, label %bb247, label %bb254

bb247:                                            ; preds = %bb242
  %1135 = load float* %r243, align 4              ; <float> [#uses=1]
  %1136 = fcmp uge float %1135, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1136, label %bb248, label %bb252

bb248:                                            ; preds = %bb247
  %1137 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1138 = getelementptr inbounds %struct.GLDContextRec* %1137, i32 0, i32 2 ; <float*> [#uses=1]
  %1139 = load float* %1138, align 4              ; <float> [#uses=1]
  %1140 = load float* %r243, align 4              ; <float> [#uses=1]
  %1141 = fcmp olt float %1139, %1140             ; <i1> [#uses=1]
  br i1 %1141, label %bb249, label %bb250

bb249:                                            ; preds = %bb248
  %1142 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1143 = getelementptr inbounds %struct.GLDContextRec* %1142, i32 0, i32 2 ; <float*> [#uses=1]
  %1144 = load float* %1143, align 4              ; <float> [#uses=1]
  store float %1144, float* %iftmp.196, align 4
  br label %bb251

bb250:                                            ; preds = %bb248
  %1145 = load float* %r243, align 4              ; <float> [#uses=1]
  store float %1145, float* %iftmp.196, align 4
  br label %bb251

bb251:                                            ; preds = %bb250, %bb249
  %1146 = load float* %iftmp.196, align 4         ; <float> [#uses=1]
  store float %1146, float* %iftmp.195, align 4
  br label %bb253

bb252:                                            ; preds = %bb247
  store float 0.000000e+00, float* %iftmp.195, align 4
  br label %bb253

bb253:                                            ; preds = %bb252, %bb251
  %1147 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1148 = getelementptr inbounds float* %1147, i32 0 ; <float*> [#uses=1]
  %1149 = load float* %iftmp.195, align 4         ; <float> [#uses=1]
  store float %1149, float* %1148, align 1
  br label %bb254

bb254:                                            ; preds = %bb253, %bb242
  %1150 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %1151 = getelementptr inbounds %struct.GLDState* %1150, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %1152 = getelementptr inbounds %struct.GLDMaskMode* %1151, i32 0, i32 3 ; <i8*> [#uses=1]
  %1153 = load i8* %1152, align 1                 ; <i8> [#uses=1]
  %1154 = zext i8 %1153 to i32                    ; <i32> [#uses=1]
  %1155 = load i32* %cur_draw_buffer_mask_bit239, align 4 ; <i32> [#uses=1]
  %1156 = and i32 %1154, %1155                    ; <i32> [#uses=1]
  %1157 = icmp ne i32 %1156, 0                    ; <i1> [#uses=1]
  br i1 %1157, label %bb255, label %bb262

bb255:                                            ; preds = %bb254
  %1158 = load float* %g244, align 4              ; <float> [#uses=1]
  %1159 = fcmp uge float %1158, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1159, label %bb256, label %bb260

bb256:                                            ; preds = %bb255
  %1160 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1161 = getelementptr inbounds %struct.GLDContextRec* %1160, i32 0, i32 2 ; <float*> [#uses=1]
  %1162 = load float* %1161, align 4              ; <float> [#uses=1]
  %1163 = load float* %g244, align 4              ; <float> [#uses=1]
  %1164 = fcmp olt float %1162, %1163             ; <i1> [#uses=1]
  br i1 %1164, label %bb257, label %bb258

bb257:                                            ; preds = %bb256
  %1165 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1166 = getelementptr inbounds %struct.GLDContextRec* %1165, i32 0, i32 2 ; <float*> [#uses=1]
  %1167 = load float* %1166, align 4              ; <float> [#uses=1]
  store float %1167, float* %iftmp.198, align 4
  br label %bb259

bb258:                                            ; preds = %bb256
  %1168 = load float* %g244, align 4              ; <float> [#uses=1]
  store float %1168, float* %iftmp.198, align 4
  br label %bb259

bb259:                                            ; preds = %bb258, %bb257
  %1169 = load float* %iftmp.198, align 4         ; <float> [#uses=1]
  store float %1169, float* %iftmp.197, align 4
  br label %bb261

bb260:                                            ; preds = %bb255
  store float 0.000000e+00, float* %iftmp.197, align 4
  br label %bb261

bb261:                                            ; preds = %bb260, %bb259
  %1170 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1171 = getelementptr inbounds float* %1170, i32 1 ; <float*> [#uses=1]
  %1172 = load float* %iftmp.197, align 4         ; <float> [#uses=1]
  store float %1172, float* %1171, align 1
  br label %bb262

bb262:                                            ; preds = %bb261, %bb254
  %1173 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %1174 = getelementptr inbounds %struct.GLDState* %1173, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %1175 = getelementptr inbounds %struct.GLDMaskMode* %1174, i32 0, i32 4 ; <i8*> [#uses=1]
  %1176 = load i8* %1175, align 2                 ; <i8> [#uses=1]
  %1177 = zext i8 %1176 to i32                    ; <i32> [#uses=1]
  %1178 = load i32* %cur_draw_buffer_mask_bit239, align 4 ; <i32> [#uses=1]
  %1179 = and i32 %1177, %1178                    ; <i32> [#uses=1]
  %1180 = icmp ne i32 %1179, 0                    ; <i1> [#uses=1]
  br i1 %1180, label %bb263, label %bb270

bb263:                                            ; preds = %bb262
  %1181 = load float* %b245, align 4              ; <float> [#uses=1]
  %1182 = fcmp uge float %1181, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1182, label %bb264, label %bb268

bb264:                                            ; preds = %bb263
  %1183 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1184 = getelementptr inbounds %struct.GLDContextRec* %1183, i32 0, i32 2 ; <float*> [#uses=1]
  %1185 = load float* %1184, align 4              ; <float> [#uses=1]
  %1186 = load float* %b245, align 4              ; <float> [#uses=1]
  %1187 = fcmp olt float %1185, %1186             ; <i1> [#uses=1]
  br i1 %1187, label %bb265, label %bb266

bb265:                                            ; preds = %bb264
  %1188 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1189 = getelementptr inbounds %struct.GLDContextRec* %1188, i32 0, i32 2 ; <float*> [#uses=1]
  %1190 = load float* %1189, align 4              ; <float> [#uses=1]
  store float %1190, float* %iftmp.200, align 4
  br label %bb267

bb266:                                            ; preds = %bb264
  %1191 = load float* %b245, align 4              ; <float> [#uses=1]
  store float %1191, float* %iftmp.200, align 4
  br label %bb267

bb267:                                            ; preds = %bb266, %bb265
  %1192 = load float* %iftmp.200, align 4         ; <float> [#uses=1]
  store float %1192, float* %iftmp.199, align 4
  br label %bb269

bb268:                                            ; preds = %bb263
  store float 0.000000e+00, float* %iftmp.199, align 4
  br label %bb269

bb269:                                            ; preds = %bb268, %bb267
  %1193 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1194 = getelementptr inbounds float* %1193, i32 2 ; <float*> [#uses=1]
  %1195 = load float* %iftmp.199, align 4         ; <float> [#uses=1]
  store float %1195, float* %1194, align 1
  br label %bb270

bb270:                                            ; preds = %bb269, %bb262
  %1196 = load %struct.GLDState** %state_addr, align 4 ; <%struct.GLDState*> [#uses=1]
  %1197 = getelementptr inbounds %struct.GLDState* %1196, i32 0, i32 23 ; <%struct.GLDMaskMode*> [#uses=1]
  %1198 = getelementptr inbounds %struct.GLDMaskMode* %1197, i32 0, i32 5 ; <i8*> [#uses=1]
  %1199 = load i8* %1198, align 1                 ; <i8> [#uses=1]
  %1200 = zext i8 %1199 to i32                    ; <i32> [#uses=1]
  %1201 = load i32* %cur_draw_buffer_mask_bit239, align 4 ; <i32> [#uses=1]
  %1202 = and i32 %1200, %1201                    ; <i32> [#uses=1]
  %1203 = icmp ne i32 %1202, 0                    ; <i1> [#uses=1]
  br i1 %1203, label %bb271, label %bb278

bb271:                                            ; preds = %bb270
  %1204 = load float* %a246, align 4              ; <float> [#uses=1]
  %1205 = fcmp uge float %1204, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1205, label %bb272, label %bb276

bb272:                                            ; preds = %bb271
  %1206 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1207 = getelementptr inbounds %struct.GLDContextRec* %1206, i32 0, i32 2 ; <float*> [#uses=1]
  %1208 = load float* %1207, align 4              ; <float> [#uses=1]
  %1209 = load float* %a246, align 4              ; <float> [#uses=1]
  %1210 = fcmp olt float %1208, %1209             ; <i1> [#uses=1]
  br i1 %1210, label %bb273, label %bb274

bb273:                                            ; preds = %bb272
  %1211 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1212 = getelementptr inbounds %struct.GLDContextRec* %1211, i32 0, i32 2 ; <float*> [#uses=1]
  %1213 = load float* %1212, align 4              ; <float> [#uses=1]
  store float %1213, float* %iftmp.202, align 4
  br label %bb275

bb274:                                            ; preds = %bb272
  %1214 = load float* %a246, align 4              ; <float> [#uses=1]
  store float %1214, float* %iftmp.202, align 4
  br label %bb275

bb275:                                            ; preds = %bb274, %bb273
  %1215 = load float* %iftmp.202, align 4         ; <float> [#uses=1]
  store float %1215, float* %iftmp.201, align 4
  br label %bb277

bb276:                                            ; preds = %bb271
  store float 0.000000e+00, float* %iftmp.201, align 4
  br label %bb277

bb277:                                            ; preds = %bb276, %bb275
  %1216 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1217 = getelementptr inbounds float* %1216, i32 3 ; <float*> [#uses=1]
  %1218 = load float* %iftmp.201, align 4         ; <float> [#uses=1]
  store float %1218, float* %1217, align 1
  br label %bb278

bb278:                                            ; preds = %bb277, %bb270
  %1219 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1220 = getelementptr inbounds double* %1219, i32 4 ; <double*> [#uses=1]
  store double* %1220, double** %accum, align 4
  %1221 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1222 = getelementptr inbounds float* %1221, i32 4 ; <float*> [#uses=1]
  store float* %1222, float** %color_ptr235, align 4
  br label %bb279

bb279:                                            ; preds = %bb278, %bb241
  %1223 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1224 = load double** %accum_end, align 4       ; <double*> [#uses=1]
  %1225 = icmp ult double* %1223, %1224           ; <i1> [#uses=1]
  br i1 %1225, label %bb242, label %bb280

bb280:                                            ; preds = %bb279
  br label %bb312

bb281:                                            ; preds = %bb240
  br label %bb311

bb282:                                            ; preds = %bb311
  %1226 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1227 = getelementptr inbounds double* %1226, i32 0 ; <double*> [#uses=1]
  %1228 = load double* %1227, align 1             ; <double> [#uses=1]
  %1229 = fptrunc double %1228 to float           ; <float> [#uses=1]
  %1230 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1231 = fmul float %1229, %1230                 ; <float> [#uses=1]
  store float %1231, float* %r283, align 4
  %1232 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1233 = getelementptr inbounds double* %1232, i32 1 ; <double*> [#uses=1]
  %1234 = load double* %1233, align 1             ; <double> [#uses=1]
  %1235 = fptrunc double %1234 to float           ; <float> [#uses=1]
  %1236 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1237 = fmul float %1235, %1236                 ; <float> [#uses=1]
  store float %1237, float* %g284, align 4
  %1238 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1239 = getelementptr inbounds double* %1238, i32 2 ; <double*> [#uses=1]
  %1240 = load double* %1239, align 1             ; <double> [#uses=1]
  %1241 = fptrunc double %1240 to float           ; <float> [#uses=1]
  %1242 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1243 = fmul float %1241, %1242                 ; <float> [#uses=1]
  store float %1243, float* %b285, align 4
  %1244 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1245 = getelementptr inbounds double* %1244, i32 3 ; <double*> [#uses=1]
  %1246 = load double* %1245, align 1             ; <double> [#uses=1]
  %1247 = fptrunc double %1246 to float           ; <float> [#uses=1]
  %1248 = load float* %value_addr, align 4        ; <float> [#uses=1]
  %1249 = fmul float %1247, %1248                 ; <float> [#uses=1]
  store float %1249, float* %a286, align 4
  %1250 = load float* %r283, align 4              ; <float> [#uses=1]
  %1251 = fcmp uge float %1250, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1251, label %bb287, label %bb291

bb287:                                            ; preds = %bb282
  %1252 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1253 = getelementptr inbounds %struct.GLDContextRec* %1252, i32 0, i32 2 ; <float*> [#uses=1]
  %1254 = load float* %1253, align 4              ; <float> [#uses=1]
  %1255 = load float* %r283, align 4              ; <float> [#uses=1]
  %1256 = fcmp olt float %1254, %1255             ; <i1> [#uses=1]
  br i1 %1256, label %bb288, label %bb289

bb288:                                            ; preds = %bb287
  %1257 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1258 = getelementptr inbounds %struct.GLDContextRec* %1257, i32 0, i32 2 ; <float*> [#uses=1]
  %1259 = load float* %1258, align 4              ; <float> [#uses=1]
  store float %1259, float* %iftmp.204, align 4
  br label %bb290

bb289:                                            ; preds = %bb287
  %1260 = load float* %r283, align 4              ; <float> [#uses=1]
  store float %1260, float* %iftmp.204, align 4
  br label %bb290

bb290:                                            ; preds = %bb289, %bb288
  %1261 = load float* %iftmp.204, align 4         ; <float> [#uses=1]
  store float %1261, float* %iftmp.203, align 4
  br label %bb292

bb291:                                            ; preds = %bb282
  store float 0.000000e+00, float* %iftmp.203, align 4
  br label %bb292

bb292:                                            ; preds = %bb291, %bb290
  %1262 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1263 = getelementptr inbounds float* %1262, i32 0 ; <float*> [#uses=1]
  %1264 = load float* %iftmp.203, align 4         ; <float> [#uses=1]
  store float %1264, float* %1263, align 1
  %1265 = load float* %g284, align 4              ; <float> [#uses=1]
  %1266 = fcmp uge float %1265, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1266, label %bb293, label %bb297

bb293:                                            ; preds = %bb292
  %1267 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1268 = getelementptr inbounds %struct.GLDContextRec* %1267, i32 0, i32 2 ; <float*> [#uses=1]
  %1269 = load float* %1268, align 4              ; <float> [#uses=1]
  %1270 = load float* %g284, align 4              ; <float> [#uses=1]
  %1271 = fcmp olt float %1269, %1270             ; <i1> [#uses=1]
  br i1 %1271, label %bb294, label %bb295

bb294:                                            ; preds = %bb293
  %1272 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1273 = getelementptr inbounds %struct.GLDContextRec* %1272, i32 0, i32 2 ; <float*> [#uses=1]
  %1274 = load float* %1273, align 4              ; <float> [#uses=1]
  store float %1274, float* %iftmp.206, align 4
  br label %bb296

bb295:                                            ; preds = %bb293
  %1275 = load float* %g284, align 4              ; <float> [#uses=1]
  store float %1275, float* %iftmp.206, align 4
  br label %bb296

bb296:                                            ; preds = %bb295, %bb294
  %1276 = load float* %iftmp.206, align 4         ; <float> [#uses=1]
  store float %1276, float* %iftmp.205, align 4
  br label %bb298

bb297:                                            ; preds = %bb292
  store float 0.000000e+00, float* %iftmp.205, align 4
  br label %bb298

bb298:                                            ; preds = %bb297, %bb296
  %1277 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1278 = getelementptr inbounds float* %1277, i32 1 ; <float*> [#uses=1]
  %1279 = load float* %iftmp.205, align 4         ; <float> [#uses=1]
  store float %1279, float* %1278, align 1
  %1280 = load float* %b285, align 4              ; <float> [#uses=1]
  %1281 = fcmp uge float %1280, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1281, label %bb299, label %bb303

bb299:                                            ; preds = %bb298
  %1282 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1283 = getelementptr inbounds %struct.GLDContextRec* %1282, i32 0, i32 2 ; <float*> [#uses=1]
  %1284 = load float* %1283, align 4              ; <float> [#uses=1]
  %1285 = load float* %b285, align 4              ; <float> [#uses=1]
  %1286 = fcmp olt float %1284, %1285             ; <i1> [#uses=1]
  br i1 %1286, label %bb300, label %bb301

bb300:                                            ; preds = %bb299
  %1287 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1288 = getelementptr inbounds %struct.GLDContextRec* %1287, i32 0, i32 2 ; <float*> [#uses=1]
  %1289 = load float* %1288, align 4              ; <float> [#uses=1]
  store float %1289, float* %iftmp.208, align 4
  br label %bb302

bb301:                                            ; preds = %bb299
  %1290 = load float* %b285, align 4              ; <float> [#uses=1]
  store float %1290, float* %iftmp.208, align 4
  br label %bb302

bb302:                                            ; preds = %bb301, %bb300
  %1291 = load float* %iftmp.208, align 4         ; <float> [#uses=1]
  store float %1291, float* %iftmp.207, align 4
  br label %bb304

bb303:                                            ; preds = %bb298
  store float 0.000000e+00, float* %iftmp.207, align 4
  br label %bb304

bb304:                                            ; preds = %bb303, %bb302
  %1292 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1293 = getelementptr inbounds float* %1292, i32 2 ; <float*> [#uses=1]
  %1294 = load float* %iftmp.207, align 4         ; <float> [#uses=1]
  store float %1294, float* %1293, align 1
  %1295 = load float* %a286, align 4              ; <float> [#uses=1]
  %1296 = fcmp uge float %1295, 0.000000e+00      ; <i1> [#uses=1]
  br i1 %1296, label %bb305, label %bb309

bb305:                                            ; preds = %bb304
  %1297 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1298 = getelementptr inbounds %struct.GLDContextRec* %1297, i32 0, i32 2 ; <float*> [#uses=1]
  %1299 = load float* %1298, align 4              ; <float> [#uses=1]
  %1300 = load float* %a286, align 4              ; <float> [#uses=1]
  %1301 = fcmp olt float %1299, %1300             ; <i1> [#uses=1]
  br i1 %1301, label %bb306, label %bb307

bb306:                                            ; preds = %bb305
  %1302 = load %struct.GLDContextRec** %ctx_addr, align 4 ; <%struct.GLDContextRec*> [#uses=1]
  %1303 = getelementptr inbounds %struct.GLDContextRec* %1302, i32 0, i32 2 ; <float*> [#uses=1]
  %1304 = load float* %1303, align 4              ; <float> [#uses=1]
  store float %1304, float* %iftmp.210, align 4
  br label %bb308

bb307:                                            ; preds = %bb305
  %1305 = load float* %a286, align 4              ; <float> [#uses=1]
  store float %1305, float* %iftmp.210, align 4
  br label %bb308

bb308:                                            ; preds = %bb307, %bb306
  %1306 = load float* %iftmp.210, align 4         ; <float> [#uses=1]
  store float %1306, float* %iftmp.209, align 4
  br label %bb310

bb309:                                            ; preds = %bb304
  store float 0.000000e+00, float* %iftmp.209, align 4
  br label %bb310

bb310:                                            ; preds = %bb309, %bb308
  %1307 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1308 = getelementptr inbounds float* %1307, i32 3 ; <float*> [#uses=1]
  %1309 = load float* %iftmp.209, align 4         ; <float> [#uses=1]
  store float %1309, float* %1308, align 1
  %1310 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1311 = getelementptr inbounds double* %1310, i32 4 ; <double*> [#uses=1]
  store double* %1311, double** %accum, align 4
  %1312 = load float** %color_ptr235, align 4     ; <float*> [#uses=1]
  %1313 = getelementptr inbounds float* %1312, i32 4 ; <float*> [#uses=1]
  store float* %1313, float** %color_ptr235, align 4
  br label %bb311

bb311:                                            ; preds = %bb310, %bb281
  %1314 = load double** %accum, align 4           ; <double*> [#uses=1]
  %1315 = load double** %accum_end, align 4       ; <double*> [#uses=1]
  %1316 = icmp ult double* %1314, %1315           ; <i1> [#uses=1]
  br i1 %1316, label %bb282, label %bb312

bb312:                                            ; preds = %bb311, %bb280
  %1317 = load i32* %y, align 4                   ; <i32> [#uses=1]
  %1318 = add i32 %1317, 1                        ; <i32> [#uses=1]
  store i32 %1318, i32* %y, align 4
  %1319 = load i32* %offset, align 4              ; <i32> [#uses=1]
  %1320 = load i32* %y_inc, align 4               ; <i32> [#uses=1]
  %1321 = add i32 %1319, %1320                    ; <i32> [#uses=1]
  store i32 %1321, i32* %offset, align 4
  br label %bb313

bb313:                                            ; preds = %bb312, %bb238
  %1322 = load i32* %y, align 4                   ; <i32> [#uses=1]
  %1323 = load i32* %yl, align 4                  ; <i32> [#uses=1]
  %1324 = icmp ult i32 %1322, %1323               ; <i1> [#uses=1]
  br i1 %1324, label %bb240, label %bb314

bb314:                                            ; preds = %bb313, %bb237
  %1325 = load i32* %draw_buffer, align 4         ; <i32> [#uses=1]
  %1326 = add nsw i32 %1325, 1                    ; <i32> [#uses=1]
  store i32 %1326, i32* %draw_buffer, align 4
  br label %bb315

bb315:                                            ; preds = %bb314, %bb234
  %1327 = load i32* %draw_buffer, align 4         ; <i32> [#uses=1]
  %1328 = icmp sle i32 %1327, 7                   ; <i1> [#uses=1]
  br i1 %1328, label %bb237, label %bb316

bb316:                                            ; preds = %bb315, %bb232, %bb228, %bb6
  br label %return

return:                                           ; preds = %bb316
  ret void
}
