; RUN: llc -mtriple=thumbv7-apple-darwin10 -mcpu=cortex-a8 < %s | FileCheck %s

; LSR should recognize that this is an unrolled loop which can use
; constant offset addressing, so that each of the following stores
; uses the same register.

; CHECK: vstr.32 s0, [r12, #-128]
; CHECK: vstr.32 s0, [r12, #-96]
; CHECK: vstr.32 s0, [r12, #-64]
; CHECK: vstr.32 s0, [r12, #-32]
; CHECK: vstr.32 s0, [r12]
; CHECK: vstr.32 s0, [r12, #32]
; CHECK: vstr.32 s0, [r12, #64]
; CHECK: vstr.32 s0, [r12, #96]

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"

%0 = type { %1*, %3*, %6*, i8*, i32, i32, %8*, i32, i32, i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8**, i32, i32, i32, i32, i32, [64 x i32]*, [4 x %9*], [4 x %10*], [4 x %10*], i32, %11*, i32, i32, [16 x i8], [16 x i8], [16 x i8], i32, i32, i8, i8, i8, i16, i16, i32, i8, i32, %12*, i32, i32, i32, i32, i8*, i32, [4 x %11*], i32, i32, i32, [10 x i32], i32, i32, i32, i32, i32, %13*, %14*, %15*, %16*, %17*, %18*, %19*, %20*, %21*, %22*, %23* }
%1 = type { void (%2*)*, void (%2*, i32)*, void (%2*)*, void (%2*, i8*)*, void (%2*)*, i32, %7, i32, i32, i8**, i32, i8**, i32, i32 }
%2 = type { %1*, %3*, %6*, i8*, i32, i32 }
%3 = type { i8* (%2*, i32, i32)*, i8* (%2*, i32, i32)*, i8** (%2*, i32, i32, i32)*, [64 x i16]** (%2*, i32, i32, i32)*, %4* (%2*, i32, i32, i32, i32, i32)*, %5* (%2*, i32, i32, i32, i32, i32)*, void (%2*)*, i8** (%2*, %4*, i32, i32, i32)*, [64 x i16]** (%2*, %5*, i32, i32, i32)*, void (%2*, i32)*, void (%2*)*, i32, i32 }
%4 = type opaque
%5 = type opaque
%6 = type { void (%2*)*, i32, i32, i32, i32 }
%7 = type { [8 x i32], [12 x i32] }
%8 = type { i8*, i32, void (%0*)*, i32 (%0*)*, void (%0*, i32)*, i32 (%0*, i32)*, void (%0*)* }
%9 = type { [64 x i16], i32 }
%10 = type { [17 x i8], [256 x i8], i32 }
%11 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %9*, i8* }
%12 = type { %12*, i8, i32, i32, i8* }
%13 = type { void (%0*)*, void (%0*)*, i32 }
%14 = type { void (%0*, i32)*, void (%0*, i8**, i32*, i32)* }
%15 = type { void (%0*)*, i32 (%0*)*, void (%0*)*, i32 (%0*, i8***)*, %5** }
%16 = type { void (%0*, i32)*, void (%0*, i8***, i32*, i32, i8**, i32*, i32)* }
%17 = type { i32 (%0*)*, void (%0*)*, void (%0*)*, void (%0*)*, i32, i32 }
%18 = type { void (%0*)*, i32 (%0*)*, i32 (%0*)*, i32, i32, i32, i32 }
%19 = type { void (%0*)*, i32 (%0*, [64 x i16]**)*, i32 }
%20 = type { void (%0*)*, [10 x void (%0*, %11*, i16*, i8**, i32)*] }
%21 = type { void (%0*)*, void (%0*, i8***, i32*, i32, i8**, i32*, i32)*, i32 }
%22 = type { void (%0*)*, void (%0*, i8***, i32, i8**, i32)* }
%23 = type { void (%0*, i32)*, void (%0*, i8**, i8**, i32)*, void (%0*)*, void (%0*)* }

define arm_apcscc void @test(%0* nocapture %a0, %11* nocapture %a1, i16* nocapture %a2, i8** nocapture %a3, i32 %a4) nounwind {
bb:
  %t = alloca [64 x float], align 4           
  %t5 = getelementptr inbounds %0* %a0, i32 0, i32 65
  %t6 = load i8** %t5, align 4              
  %t7 = getelementptr inbounds %11* %a1, i32 0, i32 20
  %t8 = load i8** %t7, align 4              
  br label %bb9

bb9:                                            
  %t10 = phi i32 [ 0, %bb ], [ %t157, %bb156 ]
  %t11 = add i32 %t10, 8                    
  %t12 = getelementptr [64 x float]* %t, i32 0, i32 %t11
  %t13 = add i32 %t10, 16                   
  %t14 = getelementptr [64 x float]* %t, i32 0, i32 %t13
  %t15 = add i32 %t10, 24                   
  %t16 = getelementptr [64 x float]* %t, i32 0, i32 %t15
  %t17 = add i32 %t10, 32                   
  %t18 = getelementptr [64 x float]* %t, i32 0, i32 %t17
  %t19 = add i32 %t10, 40                   
  %t20 = getelementptr [64 x float]* %t, i32 0, i32 %t19
  %t21 = add i32 %t10, 48                   
  %t22 = getelementptr [64 x float]* %t, i32 0, i32 %t21
  %t23 = add i32 %t10, 56                   
  %t24 = getelementptr [64 x float]* %t, i32 0, i32 %t23
  %t25 = getelementptr [64 x float]* %t, i32 0, i32 %t10
  %t26 = shl i32 %t10, 5                    
  %t27 = or i32 %t26, 8                     
  %t28 = getelementptr i8* %t8, i32 %t27  
  %t29 = bitcast i8* %t28 to float*         
  %t30 = or i32 %t26, 16                    
  %t31 = getelementptr i8* %t8, i32 %t30  
  %t32 = bitcast i8* %t31 to float*         
  %t33 = or i32 %t26, 24                    
  %t34 = getelementptr i8* %t8, i32 %t33  
  %t35 = bitcast i8* %t34 to float*         
  %t36 = or i32 %t26, 4                     
  %t37 = getelementptr i8* %t8, i32 %t36  
  %t38 = bitcast i8* %t37 to float*         
  %t39 = or i32 %t26, 12                    
  %t40 = getelementptr i8* %t8, i32 %t39  
  %t41 = bitcast i8* %t40 to float*         
  %t42 = or i32 %t26, 20                    
  %t43 = getelementptr i8* %t8, i32 %t42  
  %t44 = bitcast i8* %t43 to float*         
  %t45 = or i32 %t26, 28                    
  %t46 = getelementptr i8* %t8, i32 %t45  
  %t47 = bitcast i8* %t46 to float*         
  %t48 = getelementptr i8* %t8, i32 %t26  
  %t49 = bitcast i8* %t48 to float*         
  %t50 = shl i32 %t10, 3                    
  %t51 = or i32 %t50, 1                     
  %t52 = getelementptr i16* %a2, i32 %t51 
  %t53 = or i32 %t50, 2                     
  %t54 = getelementptr i16* %a2, i32 %t53 
  %t55 = or i32 %t50, 3                     
  %t56 = getelementptr i16* %a2, i32 %t55 
  %t57 = or i32 %t50, 4                     
  %t58 = getelementptr i16* %a2, i32 %t57 
  %t59 = or i32 %t50, 5                     
  %t60 = getelementptr i16* %a2, i32 %t59 
  %t61 = or i32 %t50, 6                     
  %t62 = getelementptr i16* %a2, i32 %t61 
  %t63 = or i32 %t50, 7                     
  %t64 = getelementptr i16* %a2, i32 %t63 
  %t65 = getelementptr i16* %a2, i32 %t50 
  %t66 = load i16* %t52, align 2            
  %t67 = icmp eq i16 %t66, 0                
  %t68 = load i16* %t54, align 2            
  %t69 = icmp eq i16 %t68, 0                
  %t70 = and i1 %t67, %t69                
  br i1 %t70, label %bb71, label %bb91

bb71:                                           
  %t72 = load i16* %t56, align 2            
  %t73 = icmp eq i16 %t72, 0                
  br i1 %t73, label %bb74, label %bb91

bb74:                                           
  %t75 = load i16* %t58, align 2            
  %t76 = icmp eq i16 %t75, 0                
  br i1 %t76, label %bb77, label %bb91

bb77:                                           
  %t78 = load i16* %t60, align 2            
  %t79 = icmp eq i16 %t78, 0                
  br i1 %t79, label %bb80, label %bb91

bb80:                                           
  %t81 = load i16* %t62, align 2            
  %t82 = icmp eq i16 %t81, 0                
  br i1 %t82, label %bb83, label %bb91

bb83:                                           
  %t84 = load i16* %t64, align 2            
  %t85 = icmp eq i16 %t84, 0                
  br i1 %t85, label %bb86, label %bb91

bb86:                                           
  %t87 = load i16* %t65, align 2            
  %t88 = sitofp i16 %t87 to float           
  %t89 = load float* %t49, align 4          
  %t90 = fmul float %t88, %t89            
  store float %t90, float* %t25, align 4
  store float %t90, float* %t12, align 4
  store float %t90, float* %t14, align 4
  store float %t90, float* %t16, align 4
  store float %t90, float* %t18, align 4
  store float %t90, float* %t20, align 4
  store float %t90, float* %t22, align 4
  store float %t90, float* %t24, align 4
  br label %bb156

bb91:                                           
  %t92 = load i16* %t65, align 2            
  %t93 = sitofp i16 %t92 to float           
  %t94 = load float* %t49, align 4          
  %t95 = fmul float %t93, %t94            
  %t96 = sitofp i16 %t68 to float           
  %t97 = load float* %t29, align 4          
  %t98 = fmul float %t96, %t97            
  %t99 = load i16* %t58, align 2            
  %t100 = sitofp i16 %t99 to float          
  %t101 = load float* %t32, align 4         
  %t102 = fmul float %t100, %t101         
  %t103 = load i16* %t62, align 2           
  %t104 = sitofp i16 %t103 to float         
  %t105 = load float* %t35, align 4         
  %t106 = fmul float %t104, %t105         
  %t107 = fadd float %t95, %t102          
  %t108 = fsub float %t95, %t102          
  %t109 = fadd float %t98, %t106          
  %t110 = fsub float %t98, %t106          
  %t111 = fmul float %t110, 0x3FF6A09E60000000
  %t112 = fsub float %t111, %t109         
  %t113 = fadd float %t107, %t109         
  %t114 = fsub float %t107, %t109         
  %t115 = fadd float %t108, %t112         
  %t116 = fsub float %t108, %t112         
  %t117 = sitofp i16 %t66 to float          
  %t118 = load float* %t38, align 4         
  %t119 = fmul float %t117, %t118         
  %t120 = load i16* %t56, align 2           
  %t121 = sitofp i16 %t120 to float         
  %t122 = load float* %t41, align 4         
  %t123 = fmul float %t121, %t122         
  %t124 = load i16* %t60, align 2           
  %t125 = sitofp i16 %t124 to float         
  %t126 = load float* %t44, align 4         
  %t127 = fmul float %t125, %t126         
  %t128 = load i16* %t64, align 2           
  %t129 = sitofp i16 %t128 to float         
  %t130 = load float* %t47, align 4         
  %t131 = fmul float %t129, %t130         
  %t132 = fadd float %t127, %t123         
  %t133 = fsub float %t127, %t123         
  %t134 = fadd float %t119, %t131         
  %t135 = fsub float %t119, %t131         
  %t136 = fadd float %t134, %t132         
  %t137 = fsub float %t134, %t132         
  %t138 = fmul float %t137, 0x3FF6A09E60000000
  %t139 = fadd float %t133, %t135         
  %t140 = fmul float %t139, 0x3FFD906BC0000000
  %t141 = fmul float %t135, 0x3FF1517A80000000
  %t142 = fsub float %t141, %t140         
  %t143 = fmul float %t133, 0xC004E7AEA0000000
  %t144 = fadd float %t143, %t140         
  %t145 = fsub float %t144, %t136         
  %t146 = fsub float %t138, %t145         
  %t147 = fadd float %t142, %t146         
  %t148 = fadd float %t113, %t136         
  store float %t148, float* %t25, align 4
  %t149 = fsub float %t113, %t136         
  store float %t149, float* %t24, align 4
  %t150 = fadd float %t115, %t145         
  store float %t150, float* %t12, align 4
  %t151 = fsub float %t115, %t145         
  store float %t151, float* %t22, align 4
  %t152 = fadd float %t116, %t146         
  store float %t152, float* %t14, align 4
  %t153 = fsub float %t116, %t146         
  store float %t153, float* %t20, align 4
  %t154 = fadd float %t114, %t147         
  store float %t154, float* %t18, align 4
  %t155 = fsub float %t114, %t147         
  store float %t155, float* %t16, align 4
  br label %bb156

bb156:                                          
  %t157 = add i32 %t10, 1                   
  %t158 = icmp eq i32 %t157, 8              
  br i1 %t158, label %bb159, label %bb9

bb159:                                          
  %t160 = add i32 %a4, 7                    
  %t161 = add i32 %a4, 1                    
  %t162 = add i32 %a4, 6                    
  %t163 = add i32 %a4, 2                    
  %t164 = add i32 %a4, 5                    
  %t165 = add i32 %a4, 4                    
  %t166 = add i32 %a4, 3                    
  br label %bb167

bb167:                                          
  %t168 = phi i32 [ 0, %bb159 ], [ %t293, %bb167 ]
  %t169 = getelementptr i8** %a3, i32 %t168
  %t170 = shl i32 %t168, 3                  
  %t171 = or i32 %t170, 4                   
  %t172 = getelementptr [64 x float]* %t, i32 0, i32 %t171
  %t173 = or i32 %t170, 2                   
  %t174 = getelementptr [64 x float]* %t, i32 0, i32 %t173
  %t175 = or i32 %t170, 6                   
  %t176 = getelementptr [64 x float]* %t, i32 0, i32 %t175
  %t177 = or i32 %t170, 5                   
  %t178 = getelementptr [64 x float]* %t, i32 0, i32 %t177
  %t179 = or i32 %t170, 3                   
  %t180 = getelementptr [64 x float]* %t, i32 0, i32 %t179
  %t181 = or i32 %t170, 1                   
  %t182 = getelementptr [64 x float]* %t, i32 0, i32 %t181
  %t183 = or i32 %t170, 7                   
  %t184 = getelementptr [64 x float]* %t, i32 0, i32 %t183
  %t185 = getelementptr [64 x float]* %t, i32 0, i32 %t170
  %t186 = load i8** %t169, align 4          
  %t187 = getelementptr inbounds i8* %t186, i32 %a4
  %t188 = load float* %t185, align 4        
  %t189 = load float* %t172, align 4        
  %t190 = fadd float %t188, %t189         
  %t191 = fsub float %t188, %t189         
  %t192 = load float* %t174, align 4        
  %t193 = load float* %t176, align 4        
  %t194 = fadd float %t192, %t193         
  %t195 = fsub float %t192, %t193         
  %t196 = fmul float %t195, 0x3FF6A09E60000000
  %t197 = fsub float %t196, %t194         
  %t198 = fadd float %t190, %t194         
  %t199 = fsub float %t190, %t194         
  %t200 = fadd float %t191, %t197         
  %t201 = fsub float %t191, %t197         
  %t202 = load float* %t178, align 4        
  %t203 = load float* %t180, align 4        
  %t204 = fadd float %t202, %t203         
  %t205 = fsub float %t202, %t203         
  %t206 = load float* %t182, align 4        
  %t207 = load float* %t184, align 4        
  %t208 = fadd float %t206, %t207         
  %t209 = fsub float %t206, %t207         
  %t210 = fadd float %t208, %t204         
  %t211 = fsub float %t208, %t204         
  %t212 = fmul float %t211, 0x3FF6A09E60000000
  %t213 = fadd float %t205, %t209         
  %t214 = fmul float %t213, 0x3FFD906BC0000000
  %t215 = fmul float %t209, 0x3FF1517A80000000
  %t216 = fsub float %t215, %t214         
  %t217 = fmul float %t205, 0xC004E7AEA0000000
  %t218 = fadd float %t217, %t214         
  %t219 = fsub float %t218, %t210         
  %t220 = fsub float %t212, %t219         
  %t221 = fadd float %t216, %t220         
  %t222 = fadd float %t198, %t210         
  %t223 = fptosi float %t222 to i32         
  %t224 = add nsw i32 %t223, 4              
  %t225 = lshr i32 %t224, 3                 
  %t226 = and i32 %t225, 1023               
  %t227 = add i32 %t226, 128                
  %t228 = getelementptr inbounds i8* %t6, i32 %t227
  %t229 = load i8* %t228, align 1           
  store i8 %t229, i8* %t187, align 1
  %t230 = fsub float %t198, %t210         
  %t231 = fptosi float %t230 to i32         
  %t232 = add nsw i32 %t231, 4              
  %t233 = lshr i32 %t232, 3                 
  %t234 = and i32 %t233, 1023               
  %t235 = add i32 %t234, 128                
  %t236 = getelementptr inbounds i8* %t6, i32 %t235
  %t237 = load i8* %t236, align 1           
  %t238 = getelementptr inbounds i8* %t186, i32 %t160
  store i8 %t237, i8* %t238, align 1
  %t239 = fadd float %t200, %t219         
  %t240 = fptosi float %t239 to i32         
  %t241 = add nsw i32 %t240, 4              
  %t242 = lshr i32 %t241, 3                 
  %t243 = and i32 %t242, 1023               
  %t244 = add i32 %t243, 128                
  %t245 = getelementptr inbounds i8* %t6, i32 %t244
  %t246 = load i8* %t245, align 1           
  %t247 = getelementptr inbounds i8* %t186, i32 %t161
  store i8 %t246, i8* %t247, align 1
  %t248 = fsub float %t200, %t219         
  %t249 = fptosi float %t248 to i32         
  %t250 = add nsw i32 %t249, 4              
  %t251 = lshr i32 %t250, 3                 
  %t252 = and i32 %t251, 1023               
  %t253 = add i32 %t252, 128                
  %t254 = getelementptr inbounds i8* %t6, i32 %t253
  %t255 = load i8* %t254, align 1           
  %t256 = getelementptr inbounds i8* %t186, i32 %t162
  store i8 %t255, i8* %t256, align 1
  %t257 = fadd float %t201, %t220         
  %t258 = fptosi float %t257 to i32         
  %t259 = add nsw i32 %t258, 4              
  %t260 = lshr i32 %t259, 3                 
  %t261 = and i32 %t260, 1023               
  %t262 = add i32 %t261, 128                
  %t263 = getelementptr inbounds i8* %t6, i32 %t262
  %t264 = load i8* %t263, align 1           
  %t265 = getelementptr inbounds i8* %t186, i32 %t163
  store i8 %t264, i8* %t265, align 1
  %t266 = fsub float %t201, %t220         
  %t267 = fptosi float %t266 to i32         
  %t268 = add nsw i32 %t267, 4              
  %t269 = lshr i32 %t268, 3                 
  %t270 = and i32 %t269, 1023               
  %t271 = add i32 %t270, 128                
  %t272 = getelementptr inbounds i8* %t6, i32 %t271
  %t273 = load i8* %t272, align 1           
  %t274 = getelementptr inbounds i8* %t186, i32 %t164
  store i8 %t273, i8* %t274, align 1
  %t275 = fadd float %t199, %t221         
  %t276 = fptosi float %t275 to i32         
  %t277 = add nsw i32 %t276, 4              
  %t278 = lshr i32 %t277, 3                 
  %t279 = and i32 %t278, 1023               
  %t280 = add i32 %t279, 128                
  %t281 = getelementptr inbounds i8* %t6, i32 %t280
  %t282 = load i8* %t281, align 1           
  %t283 = getelementptr inbounds i8* %t186, i32 %t165
  store i8 %t282, i8* %t283, align 1
  %t284 = fsub float %t199, %t221         
  %t285 = fptosi float %t284 to i32         
  %t286 = add nsw i32 %t285, 4              
  %t287 = lshr i32 %t286, 3                 
  %t288 = and i32 %t287, 1023               
  %t289 = add i32 %t288, 128                
  %t290 = getelementptr inbounds i8* %t6, i32 %t289
  %t291 = load i8* %t290, align 1           
  %t292 = getelementptr inbounds i8* %t186, i32 %t166
  store i8 %t291, i8* %t292, align 1
  %t293 = add nsw i32 %t168, 1              
  %t294 = icmp eq i32 %t293, 8              
  br i1 %t294, label %bb295, label %bb167

bb295:                                          
  ret void
}
