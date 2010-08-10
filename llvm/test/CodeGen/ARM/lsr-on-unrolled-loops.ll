; RUN: llc -mtriple=thumbv7-apple-darwin10 -mcpu=cortex-a8 < %s | FileCheck %s

; LSR should recognize that this is an unrolled loop which can use
; constant offset addressing, so that each of the following stores
; uses the same register.

; CHECK: vstr.32 s{{.*}}, [r{{.*}}, #-128]
; CHECK: vstr.32 s{{.*}}, [r{{.*}}, #-96]
; CHECK: vstr.32 s{{.*}}, [r{{.*}}, #-64]
; CHECK: vstr.32 s{{.*}}, [r{{.*}}, #-32]
; CHECK: vstr.32 s{{.*}}, [r{{.*}}]
; CHECK: vstr.32 s{{.*}}, [r{{.*}}, #32]
; CHECK: vstr.32 s{{.*}}, [r{{.*}}, #64]
; CHECK: vstr.32 s{{.*}}, [r{{.*}}, #96]

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

define void @test(%0* nocapture %a0, %11* nocapture %a1, i16* nocapture %a2, i8** nocapture %a3, i32 %a4) nounwind {
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

%struct.ct_data_s = type { %union.anon, %union.anon }
%struct.gz_header = type { i32, i32, i32, i32, i8*, i32, i32, i8*, i32, i8*, i32, i32, i32 }
%struct.internal_state = type { %struct.z_stream*, i32, i8*, i32, i8*, i32, i32, %struct.gz_header*, i32, i8, i32, i32, i32, i32, i8*, i32, i16*, i16*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [573 x %struct.ct_data_s], [61 x %struct.ct_data_s], [39 x %struct.ct_data_s], %struct.tree_desc_s, %struct.tree_desc_s, %struct.tree_desc_s, [16 x i16], [573 x i32], i32, i32, [573 x i8], i8*, i32, i32, i16*, i32, i32, i32, i32, i16, i32 }
%struct.static_tree_desc = type { i32 }
%struct.tree_desc_s = type { %struct.ct_data_s*, i32, %struct.static_tree_desc* }
%struct.z_stream = type { i8*, i32, i32, i8*, i32, i32, i8*, %struct.internal_state*, i8* (i8*, i32, i32)*, void (i8*, i8*)*, i8*, i32, i32, i32 }
%union.anon = type { i16 }

define i32 @longest_match(%struct.internal_state* %s, i32 %cur_match) nounwind optsize {
entry:
  %0 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 31 ; <i32*> [#uses=1]
  %1 = load i32* %0, align 4                      ; <i32> [#uses=2]
  %2 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 14 ; <i8**> [#uses=1]
  %3 = load i8** %2, align 4                      ; <i8*> [#uses=27]
  %4 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 27 ; <i32*> [#uses=1]
  %5 = load i32* %4, align 4                      ; <i32> [#uses=17]
  %6 = getelementptr inbounds i8* %3, i32 %5      ; <i8*> [#uses=1]
  %7 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 30 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 4                      ; <i32> [#uses=4]
  %9 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 36 ; <i32*> [#uses=1]
  %10 = load i32* %9, align 4                     ; <i32> [#uses=2]
  %11 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 11 ; <i32*> [#uses=1]
  %12 = load i32* %11, align 4                    ; <i32> [#uses=2]
  %13 = add i32 %12, -262                         ; <i32> [#uses=1]
  %14 = icmp ugt i32 %5, %13                      ; <i1> [#uses=1]
  br i1 %14, label %bb, label %bb2

bb:                                               ; preds = %entry
  %15 = add i32 %5, 262                           ; <i32> [#uses=1]
  %16 = sub i32 %15, %12                          ; <i32> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %bb, %entry
  %iftmp.48.0 = phi i32 [ %16, %bb ], [ 0, %entry ] ; <i32> [#uses=1]
  %17 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 16 ; <i16**> [#uses=1]
  %18 = load i16** %17, align 4                   ; <i16*> [#uses=1]
  %19 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 13 ; <i32*> [#uses=1]
  %20 = load i32* %19, align 4                    ; <i32> [#uses=1]
  %.sum = add i32 %5, 258                         ; <i32> [#uses=2]
  %21 = getelementptr inbounds i8* %3, i32 %.sum  ; <i8*> [#uses=1]
  %22 = add nsw i32 %5, -1                        ; <i32> [#uses=1]
  %.sum30 = add i32 %22, %8                       ; <i32> [#uses=1]
  %23 = getelementptr inbounds i8* %3, i32 %.sum30 ; <i8*> [#uses=1]
  %24 = load i8* %23, align 1                     ; <i8> [#uses=1]
  %.sum31 = add i32 %8, %5                        ; <i32> [#uses=1]
  %25 = getelementptr inbounds i8* %3, i32 %.sum31 ; <i8*> [#uses=1]
  %26 = load i8* %25, align 1                     ; <i8> [#uses=1]
  %27 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 35 ; <i32*> [#uses=1]
  %28 = load i32* %27, align 4                    ; <i32> [#uses=1]
  %29 = lshr i32 %1, 2                            ; <i32> [#uses=1]
  %30 = icmp ult i32 %8, %28                      ; <i1> [#uses=1]
  %. = select i1 %30, i32 %1, i32 %29             ; <i32> [#uses=1]
  %31 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 29 ; <i32*> [#uses=1]
  %32 = load i32* %31, align 4                    ; <i32> [#uses=4]
  %33 = icmp ugt i32 %10, %32                     ; <i1> [#uses=1]
  %nice_match.0.ph = select i1 %33, i32 %32, i32 %10 ; <i32> [#uses=1]
  %34 = getelementptr inbounds %struct.internal_state* %s, i32 0, i32 28 ; <i32*> [#uses=1]
  %35 = ptrtoint i8* %21 to i32                   ; <i32> [#uses=1]
  %36 = add nsw i32 %5, 257                       ; <i32> [#uses=1]
  %tmp81 = add i32 %., -1                         ; <i32> [#uses=1]
  br label %bb6

bb6:                                              ; preds = %bb24, %bb2
  %indvar78 = phi i32 [ 0, %bb2 ], [ %indvar.next79, %bb24 ] ; <i32> [#uses=2]
  %best_len.2 = phi i32 [ %8, %bb2 ], [ %best_len.0, %bb24 ] ; <i32> [#uses=8]
  %scan_end1.1 = phi i8 [ %24, %bb2 ], [ %scan_end1.0, %bb24 ] ; <i8> [#uses=6]
  %cur_match_addr.0 = phi i32 [ %cur_match, %bb2 ], [ %90, %bb24 ] ; <i32> [#uses=14]
  %scan_end.1 = phi i8 [ %26, %bb2 ], [ %scan_end.0, %bb24 ] ; <i8> [#uses=6]
  %37 = getelementptr inbounds i8* %3, i32 %cur_match_addr.0 ; <i8*> [#uses=1]
  %.sum32 = add i32 %cur_match_addr.0, %best_len.2 ; <i32> [#uses=1]
  %38 = getelementptr inbounds i8* %3, i32 %.sum32 ; <i8*> [#uses=1]
  %39 = load i8* %38, align 1                     ; <i8> [#uses=1]
  %40 = icmp eq i8 %39, %scan_end.1               ; <i1> [#uses=1]
  br i1 %40, label %bb7, label %bb23

bb7:                                              ; preds = %bb6
  %41 = add nsw i32 %best_len.2, -1               ; <i32> [#uses=1]
  %.sum33 = add i32 %41, %cur_match_addr.0        ; <i32> [#uses=1]
  %42 = getelementptr inbounds i8* %3, i32 %.sum33 ; <i8*> [#uses=1]
  %43 = load i8* %42, align 1                     ; <i8> [#uses=1]
  %44 = icmp eq i8 %43, %scan_end1.1              ; <i1> [#uses=1]
  br i1 %44, label %bb8, label %bb23

bb8:                                              ; preds = %bb7
  %45 = load i8* %37, align 1                     ; <i8> [#uses=1]
  %46 = load i8* %6, align 1                      ; <i8> [#uses=1]
  %47 = icmp eq i8 %45, %46                       ; <i1> [#uses=1]
  br i1 %47, label %bb9, label %bb23

bb9:                                              ; preds = %bb8
  %.sum34 = add i32 %cur_match_addr.0, 1          ; <i32> [#uses=1]
  %48 = getelementptr inbounds i8* %3, i32 %.sum34 ; <i8*> [#uses=1]
  %49 = load i8* %48, align 1                     ; <i8> [#uses=1]
  %.sum88 = add i32 %5, 1                         ; <i32> [#uses=1]
  %50 = getelementptr inbounds i8* %3, i32 %.sum88 ; <i8*> [#uses=1]
  %51 = load i8* %50, align 1                     ; <i8> [#uses=1]
  %52 = icmp eq i8 %49, %51                       ; <i1> [#uses=1]
  br i1 %52, label %bb10, label %bb23

bb10:                                             ; preds = %bb9
  %tmp39 = add i32 %cur_match_addr.0, 10          ; <i32> [#uses=1]
  %tmp41 = add i32 %cur_match_addr.0, 9           ; <i32> [#uses=1]
  %tmp44 = add i32 %cur_match_addr.0, 8           ; <i32> [#uses=1]
  %tmp47 = add i32 %cur_match_addr.0, 7           ; <i32> [#uses=1]
  %tmp50 = add i32 %cur_match_addr.0, 6           ; <i32> [#uses=1]
  %tmp53 = add i32 %cur_match_addr.0, 5           ; <i32> [#uses=1]
  %tmp56 = add i32 %cur_match_addr.0, 4           ; <i32> [#uses=1]
  %tmp59 = add i32 %cur_match_addr.0, 3           ; <i32> [#uses=1]
  br label %bb11

bb11:                                             ; preds = %bb18, %bb10
  %indvar = phi i32 [ %indvar.next, %bb18 ], [ 0, %bb10 ] ; <i32> [#uses=2]
  %tmp = shl i32 %indvar, 3                       ; <i32> [#uses=16]
  %tmp40 = add i32 %tmp39, %tmp                   ; <i32> [#uses=1]
  %scevgep = getelementptr i8* %3, i32 %tmp40     ; <i8*> [#uses=1]
  %tmp42 = add i32 %tmp41, %tmp                   ; <i32> [#uses=1]
  %scevgep43 = getelementptr i8* %3, i32 %tmp42   ; <i8*> [#uses=1]
  %tmp45 = add i32 %tmp44, %tmp                   ; <i32> [#uses=1]
  %scevgep46 = getelementptr i8* %3, i32 %tmp45   ; <i8*> [#uses=1]
  %tmp48 = add i32 %tmp47, %tmp                   ; <i32> [#uses=1]
  %scevgep49 = getelementptr i8* %3, i32 %tmp48   ; <i8*> [#uses=1]
  %tmp51 = add i32 %tmp50, %tmp                   ; <i32> [#uses=1]
  %scevgep52 = getelementptr i8* %3, i32 %tmp51   ; <i8*> [#uses=1]
  %tmp54 = add i32 %tmp53, %tmp                   ; <i32> [#uses=1]
  %scevgep55 = getelementptr i8* %3, i32 %tmp54   ; <i8*> [#uses=1]
  %tmp60 = add i32 %tmp59, %tmp                   ; <i32> [#uses=1]
  %scevgep61 = getelementptr i8* %3, i32 %tmp60   ; <i8*> [#uses=1]
  %tmp62 = add i32 %tmp, 10                       ; <i32> [#uses=1]
  %.sum89 = add i32 %5, %tmp62                    ; <i32> [#uses=2]
  %scevgep63 = getelementptr i8* %3, i32 %.sum89  ; <i8*> [#uses=2]
  %tmp64 = add i32 %tmp, 9                        ; <i32> [#uses=1]
  %.sum90 = add i32 %5, %tmp64                    ; <i32> [#uses=1]
  %scevgep65 = getelementptr i8* %3, i32 %.sum90  ; <i8*> [#uses=2]
  %tmp66 = add i32 %tmp, 8                        ; <i32> [#uses=1]
  %.sum91 = add i32 %5, %tmp66                    ; <i32> [#uses=1]
  %scevgep67 = getelementptr i8* %3, i32 %.sum91  ; <i8*> [#uses=2]
  %tmp6883 = or i32 %tmp, 7                       ; <i32> [#uses=1]
  %.sum92 = add i32 %5, %tmp6883                  ; <i32> [#uses=1]
  %scevgep69 = getelementptr i8* %3, i32 %.sum92  ; <i8*> [#uses=2]
  %tmp7084 = or i32 %tmp, 6                       ; <i32> [#uses=1]
  %.sum93 = add i32 %5, %tmp7084                  ; <i32> [#uses=1]
  %scevgep71 = getelementptr i8* %3, i32 %.sum93  ; <i8*> [#uses=2]
  %tmp7285 = or i32 %tmp, 5                       ; <i32> [#uses=1]
  %.sum94 = add i32 %5, %tmp7285                  ; <i32> [#uses=1]
  %scevgep73 = getelementptr i8* %3, i32 %.sum94  ; <i8*> [#uses=2]
  %tmp7486 = or i32 %tmp, 4                       ; <i32> [#uses=1]
  %.sum95 = add i32 %5, %tmp7486                  ; <i32> [#uses=1]
  %scevgep75 = getelementptr i8* %3, i32 %.sum95  ; <i8*> [#uses=2]
  %tmp7687 = or i32 %tmp, 3                       ; <i32> [#uses=1]
  %.sum96 = add i32 %5, %tmp7687                  ; <i32> [#uses=1]
  %scevgep77 = getelementptr i8* %3, i32 %.sum96  ; <i8*> [#uses=2]
  %53 = load i8* %scevgep77, align 1              ; <i8> [#uses=1]
  %54 = load i8* %scevgep61, align 1              ; <i8> [#uses=1]
  %55 = icmp eq i8 %53, %54                       ; <i1> [#uses=1]
  br i1 %55, label %bb12, label %bb20

bb12:                                             ; preds = %bb11
  %tmp57 = add i32 %tmp56, %tmp                   ; <i32> [#uses=1]
  %scevgep58 = getelementptr i8* %3, i32 %tmp57   ; <i8*> [#uses=1]
  %56 = load i8* %scevgep75, align 1              ; <i8> [#uses=1]
  %57 = load i8* %scevgep58, align 1              ; <i8> [#uses=1]
  %58 = icmp eq i8 %56, %57                       ; <i1> [#uses=1]
  br i1 %58, label %bb13, label %bb20

bb13:                                             ; preds = %bb12
  %59 = load i8* %scevgep73, align 1              ; <i8> [#uses=1]
  %60 = load i8* %scevgep55, align 1              ; <i8> [#uses=1]
  %61 = icmp eq i8 %59, %60                       ; <i1> [#uses=1]
  br i1 %61, label %bb14, label %bb20

bb14:                                             ; preds = %bb13
  %62 = load i8* %scevgep71, align 1              ; <i8> [#uses=1]
  %63 = load i8* %scevgep52, align 1              ; <i8> [#uses=1]
  %64 = icmp eq i8 %62, %63                       ; <i1> [#uses=1]
  br i1 %64, label %bb15, label %bb20

bb15:                                             ; preds = %bb14
  %65 = load i8* %scevgep69, align 1              ; <i8> [#uses=1]
  %66 = load i8* %scevgep49, align 1              ; <i8> [#uses=1]
  %67 = icmp eq i8 %65, %66                       ; <i1> [#uses=1]
  br i1 %67, label %bb16, label %bb20

bb16:                                             ; preds = %bb15
  %68 = load i8* %scevgep67, align 1              ; <i8> [#uses=1]
  %69 = load i8* %scevgep46, align 1              ; <i8> [#uses=1]
  %70 = icmp eq i8 %68, %69                       ; <i1> [#uses=1]
  br i1 %70, label %bb17, label %bb20

bb17:                                             ; preds = %bb16
  %71 = load i8* %scevgep65, align 1              ; <i8> [#uses=1]
  %72 = load i8* %scevgep43, align 1              ; <i8> [#uses=1]
  %73 = icmp eq i8 %71, %72                       ; <i1> [#uses=1]
  br i1 %73, label %bb18, label %bb20

bb18:                                             ; preds = %bb17
  %74 = load i8* %scevgep63, align 1              ; <i8> [#uses=1]
  %75 = load i8* %scevgep, align 1                ; <i8> [#uses=1]
  %76 = icmp eq i8 %74, %75                       ; <i1> [#uses=1]
  %77 = icmp slt i32 %.sum89, %.sum               ; <i1> [#uses=1]
  %or.cond = and i1 %76, %77                      ; <i1> [#uses=1]
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=1]
  br i1 %or.cond, label %bb11, label %bb20

bb20:                                             ; preds = %bb18, %bb17, %bb16, %bb15, %bb14, %bb13, %bb12, %bb11
  %scan.3 = phi i8* [ %scevgep77, %bb11 ], [ %scevgep75, %bb12 ], [ %scevgep73, %bb13 ], [ %scevgep71, %bb14 ], [ %scevgep69, %bb15 ], [ %scevgep67, %bb16 ], [ %scevgep65, %bb17 ], [ %scevgep63, %bb18 ] ; <i8*> [#uses=1]
  %78 = ptrtoint i8* %scan.3 to i32               ; <i32> [#uses=1]
  %79 = sub nsw i32 %78, %35                      ; <i32> [#uses=2]
  %80 = add i32 %79, 258                          ; <i32> [#uses=5]
  %81 = icmp sgt i32 %80, %best_len.2             ; <i1> [#uses=1]
  br i1 %81, label %bb21, label %bb23

bb21:                                             ; preds = %bb20
  store i32 %cur_match_addr.0, i32* %34, align 4
  %82 = icmp slt i32 %80, %nice_match.0.ph        ; <i1> [#uses=1]
  br i1 %82, label %bb22, label %bb25

bb22:                                             ; preds = %bb21
  %.sum37 = add i32 %36, %79                      ; <i32> [#uses=1]
  %83 = getelementptr inbounds i8* %3, i32 %.sum37 ; <i8*> [#uses=1]
  %84 = load i8* %83, align 1                     ; <i8> [#uses=1]
  %.sum38 = add i32 %80, %5                       ; <i32> [#uses=1]
  %85 = getelementptr inbounds i8* %3, i32 %.sum38 ; <i8*> [#uses=1]
  %86 = load i8* %85, align 1                     ; <i8> [#uses=1]
  br label %bb23

bb23:                                             ; preds = %bb22, %bb20, %bb9, %bb8, %bb7, %bb6
  %best_len.0 = phi i32 [ %best_len.2, %bb6 ], [ %best_len.2, %bb7 ], [ %best_len.2, %bb8 ], [ %best_len.2, %bb9 ], [ %80, %bb22 ], [ %best_len.2, %bb20 ] ; <i32> [#uses=3]
  %scan_end1.0 = phi i8 [ %scan_end1.1, %bb6 ], [ %scan_end1.1, %bb7 ], [ %scan_end1.1, %bb8 ], [ %scan_end1.1, %bb9 ], [ %84, %bb22 ], [ %scan_end1.1, %bb20 ] ; <i8> [#uses=1]
  %scan_end.0 = phi i8 [ %scan_end.1, %bb6 ], [ %scan_end.1, %bb7 ], [ %scan_end.1, %bb8 ], [ %scan_end.1, %bb9 ], [ %86, %bb22 ], [ %scan_end.1, %bb20 ] ; <i8> [#uses=1]
  %87 = and i32 %cur_match_addr.0, %20            ; <i32> [#uses=1]
  %88 = getelementptr inbounds i16* %18, i32 %87  ; <i16*> [#uses=1]
  %89 = load i16* %88, align 2                    ; <i16> [#uses=1]
  %90 = zext i16 %89 to i32                       ; <i32> [#uses=2]
  %91 = icmp ugt i32 %90, %iftmp.48.0             ; <i1> [#uses=1]
  br i1 %91, label %bb24, label %bb25

bb24:                                             ; preds = %bb23

; LSR should use count-down iteration to avoid requiring the trip count
; in a register, and it shouldn't require any reloads here.

;      CHECK: @ %bb24
; CHECK-NEXT: @   in Loop: Header=BB1_1 Depth=1
; CHECK-NEXT: sub{{.*}} [[REGISTER:r[0-9]+]], #1
; CHECK-NEXT: bne.w

  %92 = icmp eq i32 %tmp81, %indvar78             ; <i1> [#uses=1]
  %indvar.next79 = add i32 %indvar78, 1           ; <i32> [#uses=1]
  br i1 %92, label %bb25, label %bb6

bb25:                                             ; preds = %bb24, %bb23, %bb21
  %best_len.1 = phi i32 [ %best_len.0, %bb23 ], [ %best_len.0, %bb24 ], [ %80, %bb21 ] ; <i32> [#uses=2]
  %93 = icmp ugt i32 %best_len.1, %32             ; <i1> [#uses=1]
  %merge = select i1 %93, i32 %32, i32 %best_len.1 ; <i32> [#uses=1]
  ret i32 %merge
}
