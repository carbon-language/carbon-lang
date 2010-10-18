; RUN: opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output < %s |& FileCheck  %s

declare void @callee(double* %callee_arg)
declare void @nocap_callee(double* nocapture %nocap_callee_arg)

declare double* @normal_returner()
declare noalias double* @noalias_returner()

define void @caller_a(double* %arg_a0,
                      double* %arg_a1,
                      double* noalias %noalias_arg_a0,
                      double* noalias %noalias_arg_a1,
                      double** %indirect_a0,
                      double** %indirect_a1) {
  %loaded_a0 = load double** %indirect_a0
  %loaded_a1 = load double** %indirect_a1

  %escape_alloca_a0 = alloca double
  %escape_alloca_a1 = alloca double
  %noescape_alloca_a0 = alloca double
  %noescape_alloca_a1 = alloca double

  %normal_ret_a0 = call double* @normal_returner()
  %normal_ret_a1 = call double* @normal_returner()
  %noalias_ret_a0 = call double* @noalias_returner()
  %noalias_ret_a1 = call double* @noalias_returner()

  call void @callee(double* %escape_alloca_a0)
  call void @callee(double* %escape_alloca_a1)
  call void @nocap_callee(double* %noescape_alloca_a0)
  call void @nocap_callee(double* %noescape_alloca_a1)

  store double 0.0, double* %loaded_a0
  store double 0.0, double* %loaded_a1
  store double 0.0, double* %arg_a0
  store double 0.0, double* %arg_a1
  store double 0.0, double* %noalias_arg_a0
  store double 0.0, double* %noalias_arg_a1
  store double 0.0, double* %escape_alloca_a0
  store double 0.0, double* %escape_alloca_a1
  store double 0.0, double* %noescape_alloca_a0
  store double 0.0, double* %noescape_alloca_a1
  store double 0.0, double* %normal_ret_a0
  store double 0.0, double* %normal_ret_a1
  store double 0.0, double* %noalias_ret_a0
  store double 0.0, double* %noalias_ret_a1
  ret void
}

; CHECK: Function: caller_a: 16 pointers, 8 call sites
; CHECK:   MayAlias:	double* %arg_a0, double* %arg_a1
; CHECK:   NoAlias:	double* %arg_a0, double* %noalias_arg_a0
; CHECK:   NoAlias:	double* %arg_a1, double* %noalias_arg_a0
; CHECK:   NoAlias:	double* %arg_a0, double* %noalias_arg_a1
; CHECK:   NoAlias:	double* %arg_a1, double* %noalias_arg_a1
; CHECK:   NoAlias:	double* %noalias_arg_a0, double* %noalias_arg_a1
; CHECK:   MayAlias:	double* %arg_a0, double** %indirect_a0
; CHECK:   MayAlias:	double* %arg_a1, double** %indirect_a0
; CHECK:   NoAlias:	double* %noalias_arg_a0, double** %indirect_a0
; CHECK:   NoAlias:	double* %noalias_arg_a1, double** %indirect_a0
; CHECK:   MayAlias:	double* %arg_a0, double** %indirect_a1
; CHECK:   MayAlias:	double* %arg_a1, double** %indirect_a1
; CHECK:   NoAlias:	double* %noalias_arg_a0, double** %indirect_a1
; CHECK:   NoAlias:	double* %noalias_arg_a1, double** %indirect_a1
; CHECK:   MayAlias:	double** %indirect_a0, double** %indirect_a1
; CHECK:   MayAlias:	double* %arg_a0, double* %loaded_a0
; CHECK:   MayAlias:	double* %arg_a1, double* %loaded_a0
; CHECK:   NoAlias:	double* %loaded_a0, double* %noalias_arg_a0
; CHECK:   NoAlias:	double* %loaded_a0, double* %noalias_arg_a1
; CHECK:   MayAlias:	double* %loaded_a0, double** %indirect_a0
; CHECK:   MayAlias:	double* %loaded_a0, double** %indirect_a1
; CHECK:   MayAlias:	double* %arg_a0, double* %loaded_a1
; CHECK:   MayAlias:	double* %arg_a1, double* %loaded_a1
; CHECK:   NoAlias:	double* %loaded_a1, double* %noalias_arg_a0
; CHECK:   NoAlias:	double* %loaded_a1, double* %noalias_arg_a1
; CHECK:   MayAlias:	double* %loaded_a1, double** %indirect_a0
; CHECK:   MayAlias:	double* %loaded_a1, double** %indirect_a1
; CHECK:   MayAlias:	double* %loaded_a0, double* %loaded_a1
; CHECK:   NoAlias:	double* %arg_a0, double* %escape_alloca_a0
; CHECK:   NoAlias:	double* %arg_a1, double* %escape_alloca_a0
; CHECK:   NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a0
; CHECK:   NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a1
; CHECK:   NoAlias:	double* %escape_alloca_a0, double** %indirect_a0
; CHECK:   NoAlias:	double* %escape_alloca_a0, double** %indirect_a1
; CHECK:   MayAlias:	double* %escape_alloca_a0, double* %loaded_a0
; CHECK:   MayAlias:	double* %escape_alloca_a0, double* %loaded_a1
; CHECK:   NoAlias:	double* %arg_a0, double* %escape_alloca_a1
; CHECK:   NoAlias:	double* %arg_a1, double* %escape_alloca_a1
; CHECK:   NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a0
; CHECK:   NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a1
; CHECK:   NoAlias:	double* %escape_alloca_a1, double** %indirect_a0
; CHECK:   NoAlias:	double* %escape_alloca_a1, double** %indirect_a1
; CHECK:   MayAlias:	double* %escape_alloca_a1, double* %loaded_a0
; CHECK:   MayAlias:	double* %escape_alloca_a1, double* %loaded_a1
; CHECK:   NoAlias:	double* %escape_alloca_a0, double* %escape_alloca_a1
; CHECK:   NoAlias:	double* %arg_a0, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %arg_a1, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %noescape_alloca_a0, double** %indirect_a0
; CHECK:   NoAlias:	double* %noescape_alloca_a0, double** %indirect_a1
; CHECK:   NoAlias:	double* %loaded_a0, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %loaded_a1, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %arg_a0, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %arg_a1, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %noescape_alloca_a1, double** %indirect_a0
; CHECK:   NoAlias:	double* %noescape_alloca_a1, double** %indirect_a1
; CHECK:   NoAlias:	double* %loaded_a0, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %loaded_a1, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %noescape_alloca_a0, double* %noescape_alloca_a1
; CHECK:   MayAlias:	double* %arg_a0, double* %normal_ret_a0
; CHECK:   MayAlias:	double* %arg_a1, double* %normal_ret_a0
; CHECK:   NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a0
; CHECK:   NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a0
; CHECK:   MayAlias:	double* %normal_ret_a0, double** %indirect_a0
; CHECK:   MayAlias:	double* %normal_ret_a0, double** %indirect_a1
; CHECK:   MayAlias:	double* %loaded_a0, double* %normal_ret_a0
; CHECK:   MayAlias:	double* %loaded_a1, double* %normal_ret_a0
; CHECK:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a0
; CHECK:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a0
; CHECK:   NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a0
; CHECK:   NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a0
; CHECK:   MayAlias:	double* %arg_a0, double* %normal_ret_a1
; CHECK:   MayAlias:	double* %arg_a1, double* %normal_ret_a1
; CHECK:   NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a1
; CHECK:   NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a1
; CHECK:   MayAlias:	double* %normal_ret_a1, double** %indirect_a0
; CHECK:   MayAlias:	double* %normal_ret_a1, double** %indirect_a1
; CHECK:   MayAlias:	double* %loaded_a0, double* %normal_ret_a1
; CHECK:   MayAlias:	double* %loaded_a1, double* %normal_ret_a1
; CHECK:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a1
; CHECK:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a1
; CHECK:   NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a1
; CHECK:   NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a1
; CHECK:   MayAlias:	double* %normal_ret_a0, double* %normal_ret_a1
; CHECK:   NoAlias:	double* %arg_a0, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %arg_a1, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %noalias_ret_a0, double** %indirect_a0
; CHECK:   NoAlias:	double* %noalias_ret_a0, double** %indirect_a1
; CHECK:   NoAlias:	double* %loaded_a0, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %loaded_a1, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a0
; CHECK:   NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a0
; CHECK:   NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a1
; CHECK:   NoAlias:	double* %arg_a0, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %arg_a1, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %noalias_ret_a1, double** %indirect_a0
; CHECK:   NoAlias:	double* %noalias_ret_a1, double** %indirect_a1
; CHECK:   NoAlias:	double* %loaded_a0, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %loaded_a1, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a1
; CHECK:   NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a0
; CHECK:   NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a1
; CHECK:   NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a0
; CHECK:   NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a1
; CHECK:   NoAlias:	double* %noalias_ret_a0, double* %noalias_ret_a1
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a0 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a1 = call double* @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a1 = call double* @normal_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a1 = call double* @normal_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a1 = call double* @normal_returner() 
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() 
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; CHECK: ===== Alias Analysis Evaluator Report =====
; CHECK:   120 Total Alias Queries Performed
; CHECK:   84 no alias responses (70.0%)
; CHECK:   36 may alias responses (30.0%)
; CHECK:   0 must alias responses (0.0%)
; CHECK:   Alias Analysis Evaluator Pointer Alias Summary: 70%/30%/0%
; CHECK:   184 Total ModRef Queries Performed
; CHECK:   44 no mod/ref responses (23.9%)
; CHECK:   0 mod responses (0.0%)
; CHECK:   0 ref responses (0.0%)
; CHECK:   140 mod & ref responses (76.0%)
; CHECK:   Alias Analysis Evaluator Mod/Ref Summary: 23%/0%/0%/76%
