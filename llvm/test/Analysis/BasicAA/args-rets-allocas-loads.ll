; RUN: opt -interprocedural-basic-aa -interprocedural-aa-eval -print-all-alias-modref-info -disable-output < %s |& FileCheck --check-prefix=IPAA %s
; RUN: opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output < %s |& FileCheck --check-prefix=FUNCAA %s

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

; caller_b is the same as caller_a but with different names, to test
; interprocedural queries.
define void @caller_b(double* %arg_b0,
                      double* %arg_b1,
                      double* noalias %noalias_arg_b0,
                      double* noalias %noalias_arg_b1,
                      double** %indirect_b0,
                      double** %indirect_b1) {
  %loaded_b0 = load double** %indirect_b0
  %loaded_b1 = load double** %indirect_b1

  %escape_alloca_b0 = alloca double
  %escape_alloca_b1 = alloca double
  %noescape_alloca_b0 = alloca double
  %noescape_alloca_b1 = alloca double

  %normal_ret_b0 = call double* @normal_returner()
  %normal_ret_b1 = call double* @normal_returner()
  %noalias_ret_b0 = call double* @noalias_returner()
  %noalias_ret_b1 = call double* @noalias_returner()

  call void @callee(double* %escape_alloca_b0)
  call void @callee(double* %escape_alloca_b1)
  call void @nocap_callee(double* %noescape_alloca_b0)
  call void @nocap_callee(double* %noescape_alloca_b1)

  store double 0.0, double* %loaded_b0
  store double 0.0, double* %loaded_b1
  store double 0.0, double* %arg_b0
  store double 0.0, double* %arg_b1
  store double 0.0, double* %noalias_arg_b0
  store double 0.0, double* %noalias_arg_b1
  store double 0.0, double* %escape_alloca_b0
  store double 0.0, double* %escape_alloca_b1
  store double 0.0, double* %noescape_alloca_b0
  store double 0.0, double* %noescape_alloca_b1
  store double 0.0, double* %normal_ret_b0
  store double 0.0, double* %normal_ret_b1
  store double 0.0, double* %noalias_ret_b0
  store double 0.0, double* %noalias_ret_b1
  ret void
}

; FUNCAA: Function: caller_a: 16 pointers, 8 call sites
; FUNCAA:   MayAlias:	double* %arg_a0, double* %arg_a1
; FUNCAA:   NoAlias:	double* %arg_a0, double* %noalias_arg_a0
; FUNCAA:   NoAlias:	double* %arg_a1, double* %noalias_arg_a0
; FUNCAA:   NoAlias:	double* %arg_a0, double* %noalias_arg_a1
; FUNCAA:   NoAlias:	double* %arg_a1, double* %noalias_arg_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double* %noalias_arg_a1
; FUNCAA:   MayAlias:	double* %arg_a0, double** %indirect_a0
; FUNCAA:   MayAlias:	double* %arg_a1, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double** %indirect_a0
; FUNCAA:   MayAlias:	double* %arg_a0, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %arg_a1, double** %indirect_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double** %indirect_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double** %indirect_a1
; FUNCAA:   MayAlias:	double** %indirect_a0, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %arg_a0, double* %loaded_a0
; FUNCAA:   MayAlias:	double* %arg_a1, double* %loaded_a0
; FUNCAA:   NoAlias:	double* %loaded_a0, double* %noalias_arg_a0
; FUNCAA:   NoAlias:	double* %loaded_a0, double* %noalias_arg_a1
; FUNCAA:   MayAlias:	double* %loaded_a0, double** %indirect_a0
; FUNCAA:   MayAlias:	double* %loaded_a0, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %arg_a0, double* %loaded_a1
; FUNCAA:   MayAlias:	double* %arg_a1, double* %loaded_a1
; FUNCAA:   NoAlias:	double* %loaded_a1, double* %noalias_arg_a0
; FUNCAA:   NoAlias:	double* %loaded_a1, double* %noalias_arg_a1
; FUNCAA:   MayAlias:	double* %loaded_a1, double** %indirect_a0
; FUNCAA:   MayAlias:	double* %loaded_a1, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %loaded_a0, double* %loaded_a1
; FUNCAA:   NoAlias:	double* %arg_a0, double* %escape_alloca_a0
; FUNCAA:   NoAlias:	double* %arg_a1, double* %escape_alloca_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %escape_alloca_a0, double* %loaded_a0
; FUNCAA:   MayAlias:	double* %escape_alloca_a0, double* %loaded_a1
; FUNCAA:   NoAlias:	double* %arg_a0, double* %escape_alloca_a1
; FUNCAA:   NoAlias:	double* %arg_a1, double* %escape_alloca_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %escape_alloca_a1, double* %loaded_a0
; FUNCAA:   MayAlias:	double* %escape_alloca_a1, double* %loaded_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double* %escape_alloca_a1
; FUNCAA:   NoAlias:	double* %arg_a0, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %arg_a1, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %noescape_alloca_a0, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %noescape_alloca_a0, double** %indirect_a1
; FUNCAA:   NoAlias:	double* %loaded_a0, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %loaded_a1, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %arg_a0, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %arg_a1, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %noescape_alloca_a1, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %noescape_alloca_a1, double** %indirect_a1
; FUNCAA:   NoAlias:	double* %loaded_a0, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %loaded_a1, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %noescape_alloca_a0, double* %noescape_alloca_a1
; FUNCAA:   MayAlias:	double* %arg_a0, double* %normal_ret_a0
; FUNCAA:   MayAlias:	double* %arg_a1, double* %normal_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a0
; FUNCAA:   MayAlias:	double* %normal_ret_a0, double** %indirect_a0
; FUNCAA:   MayAlias:	double* %normal_ret_a0, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %loaded_a0, double* %normal_ret_a0
; FUNCAA:   MayAlias:	double* %loaded_a1, double* %normal_ret_a0
; FUNCAA:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a0
; FUNCAA:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a0
; FUNCAA:   NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a0
; FUNCAA:   NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a0
; FUNCAA:   MayAlias:	double* %arg_a0, double* %normal_ret_a1
; FUNCAA:   MayAlias:	double* %arg_a1, double* %normal_ret_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a1
; FUNCAA:   MayAlias:	double* %normal_ret_a1, double** %indirect_a0
; FUNCAA:   MayAlias:	double* %normal_ret_a1, double** %indirect_a1
; FUNCAA:   MayAlias:	double* %loaded_a0, double* %normal_ret_a1
; FUNCAA:   MayAlias:	double* %loaded_a1, double* %normal_ret_a1
; FUNCAA:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a1
; FUNCAA:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a1
; FUNCAA:   NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a1
; FUNCAA:   NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a1
; FUNCAA:   MayAlias:	double* %normal_ret_a0, double* %normal_ret_a1
; FUNCAA:   NoAlias:	double* %arg_a0, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %arg_a1, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a0, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a0, double** %indirect_a1
; FUNCAA:   NoAlias:	double* %loaded_a0, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %loaded_a1, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a1
; FUNCAA:   NoAlias:	double* %arg_a0, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %arg_a1, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %noalias_ret_a1, double** %indirect_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a1, double** %indirect_a1
; FUNCAA:   NoAlias:	double* %loaded_a0, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %loaded_a1, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a1
; FUNCAA:   NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a1
; FUNCAA:   NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a0
; FUNCAA:   NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a1
; FUNCAA:   NoAlias:	double* %noalias_ret_a0, double* %noalias_ret_a1
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; FUNCAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; FUNCAA: Function: caller_b: 16 pointers, 8 call sites
; FUNCAA:   MayAlias:	double* %arg_b0, double* %arg_b1
; FUNCAA:   NoAlias:	double* %arg_b0, double* %noalias_arg_b0
; FUNCAA:   NoAlias:	double* %arg_b1, double* %noalias_arg_b0
; FUNCAA:   NoAlias:	double* %arg_b0, double* %noalias_arg_b1
; FUNCAA:   NoAlias:	double* %arg_b1, double* %noalias_arg_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double* %noalias_arg_b1
; FUNCAA:   MayAlias:	double* %arg_b0, double** %indirect_b0
; FUNCAA:   MayAlias:	double* %arg_b1, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double** %indirect_b0
; FUNCAA:   MayAlias:	double* %arg_b0, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %arg_b1, double** %indirect_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double** %indirect_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double** %indirect_b1
; FUNCAA:   MayAlias:	double** %indirect_b0, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %arg_b0, double* %loaded_b0
; FUNCAA:   MayAlias:	double* %arg_b1, double* %loaded_b0
; FUNCAA:   NoAlias:	double* %loaded_b0, double* %noalias_arg_b0
; FUNCAA:   NoAlias:	double* %loaded_b0, double* %noalias_arg_b1
; FUNCAA:   MayAlias:	double* %loaded_b0, double** %indirect_b0
; FUNCAA:   MayAlias:	double* %loaded_b0, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %arg_b0, double* %loaded_b1
; FUNCAA:   MayAlias:	double* %arg_b1, double* %loaded_b1
; FUNCAA:   NoAlias:	double* %loaded_b1, double* %noalias_arg_b0
; FUNCAA:   NoAlias:	double* %loaded_b1, double* %noalias_arg_b1
; FUNCAA:   MayAlias:	double* %loaded_b1, double** %indirect_b0
; FUNCAA:   MayAlias:	double* %loaded_b1, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %loaded_b0, double* %loaded_b1
; FUNCAA:   NoAlias:	double* %arg_b0, double* %escape_alloca_b0
; FUNCAA:   NoAlias:	double* %arg_b1, double* %escape_alloca_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_arg_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_arg_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %escape_alloca_b0, double* %loaded_b0
; FUNCAA:   MayAlias:	double* %escape_alloca_b0, double* %loaded_b1
; FUNCAA:   NoAlias:	double* %arg_b0, double* %escape_alloca_b1
; FUNCAA:   NoAlias:	double* %arg_b1, double* %escape_alloca_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_arg_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_arg_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %escape_alloca_b1, double* %loaded_b0
; FUNCAA:   MayAlias:	double* %escape_alloca_b1, double* %loaded_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double* %escape_alloca_b1
; FUNCAA:   NoAlias:	double* %arg_b0, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %arg_b1, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %noescape_alloca_b0, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %noescape_alloca_b0, double** %indirect_b1
; FUNCAA:   NoAlias:	double* %loaded_b0, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %loaded_b1, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %arg_b0, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %arg_b1, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %noescape_alloca_b1, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %noescape_alloca_b1, double** %indirect_b1
; FUNCAA:   NoAlias:	double* %loaded_b0, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %loaded_b1, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %noescape_alloca_b0, double* %noescape_alloca_b1
; FUNCAA:   MayAlias:	double* %arg_b0, double* %normal_ret_b0
; FUNCAA:   MayAlias:	double* %arg_b1, double* %normal_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double* %normal_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double* %normal_ret_b0
; FUNCAA:   MayAlias:	double* %normal_ret_b0, double** %indirect_b0
; FUNCAA:   MayAlias:	double* %normal_ret_b0, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %loaded_b0, double* %normal_ret_b0
; FUNCAA:   MayAlias:	double* %loaded_b1, double* %normal_ret_b0
; FUNCAA:   MayAlias:	double* %escape_alloca_b0, double* %normal_ret_b0
; FUNCAA:   MayAlias:	double* %escape_alloca_b1, double* %normal_ret_b0
; FUNCAA:   NoAlias:	double* %noescape_alloca_b0, double* %normal_ret_b0
; FUNCAA:   NoAlias:	double* %noescape_alloca_b1, double* %normal_ret_b0
; FUNCAA:   MayAlias:	double* %arg_b0, double* %normal_ret_b1
; FUNCAA:   MayAlias:	double* %arg_b1, double* %normal_ret_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double* %normal_ret_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double* %normal_ret_b1
; FUNCAA:   MayAlias:	double* %normal_ret_b1, double** %indirect_b0
; FUNCAA:   MayAlias:	double* %normal_ret_b1, double** %indirect_b1
; FUNCAA:   MayAlias:	double* %loaded_b0, double* %normal_ret_b1
; FUNCAA:   MayAlias:	double* %loaded_b1, double* %normal_ret_b1
; FUNCAA:   MayAlias:	double* %escape_alloca_b0, double* %normal_ret_b1
; FUNCAA:   MayAlias:	double* %escape_alloca_b1, double* %normal_ret_b1
; FUNCAA:   NoAlias:	double* %noescape_alloca_b0, double* %normal_ret_b1
; FUNCAA:   NoAlias:	double* %noescape_alloca_b1, double* %normal_ret_b1
; FUNCAA:   MayAlias:	double* %normal_ret_b0, double* %normal_ret_b1
; FUNCAA:   NoAlias:	double* %arg_b0, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %arg_b1, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b0, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b0, double** %indirect_b1
; FUNCAA:   NoAlias:	double* %loaded_b0, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %loaded_b1, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b0, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b0, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %noalias_ret_b0, double* %normal_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b0, double* %normal_ret_b1
; FUNCAA:   NoAlias:	double* %arg_b0, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %arg_b1, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b0, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %noalias_arg_b1, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %noalias_ret_b1, double** %indirect_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b1, double** %indirect_b1
; FUNCAA:   NoAlias:	double* %loaded_b0, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %loaded_b1, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_ret_b1
; FUNCAA:   NoAlias:	double* %noalias_ret_b1, double* %noescape_alloca_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b1, double* %noescape_alloca_b1
; FUNCAA:   NoAlias:	double* %noalias_ret_b1, double* %normal_ret_b0
; FUNCAA:   NoAlias:	double* %noalias_ret_b1, double* %normal_ret_b1
; FUNCAA:   NoAlias:	double* %noalias_ret_b0, double* %noalias_ret_b1
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @callee(double* %escape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @callee(double* %escape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; FUNCAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; FUNCAA: ===== Alias Analysis Evaluator Report =====
; FUNCAA:   240 Total Alias Queries Performed
; FUNCAA:   168 no alias responses (70.0%)
; FUNCAA:   72 may alias responses (30.0%)
; FUNCAA:   0 must alias responses (0.0%)
; FUNCAA:   Alias Analysis Evaluator Pointer Alias Summary: 70%/30%/0%
; FUNCAA:   256 Total ModRef Queries Performed
; FUNCAA:   88 no mod/ref responses (34.3%)
; FUNCAA:   0 mod responses (0.0%)
; FUNCAA:   0 ref responses (0.0%)
; FUNCAA:   168 mod & ref responses (65.6%)
; FUNCAA:   Alias Analysis Evaluator Mod/Ref Summary: 34%/0%/0%/65%

; IPAA: Module: 34 pointers, 16 call sites
; IPAA:   MayAlias:	double* %callee_arg, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a1, double* %callee_arg
; IPAA:   MayAlias:	double* %arg_a1, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %arg_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %nocap_callee_arg
; IPAA:   NoAlias:	double* %arg_a0, double* %noalias_arg_a0
; IPAA:   NoAlias:	double* %arg_a1, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %nocap_callee_arg
; IPAA:   NoAlias:	double* %arg_a0, double* %noalias_arg_a1
; IPAA:   NoAlias:	double* %arg_a1, double* %noalias_arg_a1
; IPAA:   NoAlias:	double* %noalias_arg_a0, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %callee_arg, double** %indirect_a0
; IPAA:   MayAlias:	double* %nocap_callee_arg, double** %indirect_a0
; IPAA:   MayAlias:	double* %arg_a0, double** %indirect_a0
; IPAA:   MayAlias:	double* %arg_a1, double** %indirect_a0
; IPAA:   NoAlias:	double* %noalias_arg_a0, double** %indirect_a0
; IPAA:   NoAlias:	double* %noalias_arg_a1, double** %indirect_a0
; IPAA:   MayAlias:	double* %callee_arg, double** %indirect_a1
; IPAA:   MayAlias:	double* %nocap_callee_arg, double** %indirect_a1
; IPAA:   MayAlias:	double* %arg_a0, double** %indirect_a1
; IPAA:   MayAlias:	double* %arg_a1, double** %indirect_a1
; IPAA:   NoAlias:	double* %noalias_arg_a0, double** %indirect_a1
; IPAA:   NoAlias:	double* %noalias_arg_a1, double** %indirect_a1
; IPAA:   MayAlias:	double** %indirect_a0, double** %indirect_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %loaded_a0
; IPAA:   MayAlias:	double* %loaded_a0, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %loaded_a0
; IPAA:   MayAlias:	double* %arg_a1, double* %loaded_a0
; IPAA:   NoAlias:	double* %loaded_a0, double* %noalias_arg_a0
; IPAA:   NoAlias:	double* %loaded_a0, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %loaded_a0, double** %indirect_a0
; IPAA:   MayAlias:	double* %loaded_a0, double** %indirect_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %loaded_a1
; IPAA:   MayAlias:	double* %loaded_a1, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %loaded_a1
; IPAA:   MayAlias:	double* %arg_a1, double* %loaded_a1
; IPAA:   NoAlias:	double* %loaded_a1, double* %noalias_arg_a0
; IPAA:   NoAlias:	double* %loaded_a1, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %loaded_a1, double** %indirect_a0
; IPAA:   MayAlias:	double* %loaded_a1, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %loaded_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %escape_alloca_a0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %nocap_callee_arg
; IPAA:   NoAlias:	double* %arg_a0, double* %escape_alloca_a0
; IPAA:   NoAlias:	double* %arg_a1, double* %escape_alloca_a0
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a0
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a1
; IPAA:   NoAlias:	double* %escape_alloca_a0, double** %indirect_a0
; IPAA:   NoAlias:	double* %escape_alloca_a0, double** %indirect_a1
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %loaded_a0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %loaded_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %escape_alloca_a1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %nocap_callee_arg
; IPAA:   NoAlias:	double* %arg_a0, double* %escape_alloca_a1
; IPAA:   NoAlias:	double* %arg_a1, double* %escape_alloca_a1
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a0
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a1
; IPAA:   NoAlias:	double* %escape_alloca_a1, double** %indirect_a0
; IPAA:   NoAlias:	double* %escape_alloca_a1, double** %indirect_a1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %loaded_a0
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %loaded_a1
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %escape_alloca_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %arg_a0, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %arg_a1, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %noescape_alloca_a0, double** %indirect_a0
; IPAA:   NoAlias:	double* %noescape_alloca_a0, double** %indirect_a1
; IPAA:   NoAlias:	double* %loaded_a0, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %loaded_a1, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %callee_arg, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %arg_a0, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %arg_a1, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %noescape_alloca_a1, double** %indirect_a0
; IPAA:   NoAlias:	double* %noescape_alloca_a1, double** %indirect_a1
; IPAA:   NoAlias:	double* %loaded_a0, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %loaded_a1, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %noescape_alloca_a0, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %arg_a0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %arg_a1, double* %normal_ret_a0
; IPAA:   NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a0
; IPAA:   NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %normal_ret_a0, double** %indirect_a0
; IPAA:   MayAlias:	double* %normal_ret_a0, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %loaded_a1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a0
; IPAA:   NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a0
; IPAA:   NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %callee_arg, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %arg_a0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %arg_a1, double* %normal_ret_a1
; IPAA:   NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a1
; IPAA:   NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %normal_ret_a1, double** %indirect_a0
; IPAA:   MayAlias:	double* %normal_ret_a1, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %loaded_a1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a1
; IPAA:   NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a1
; IPAA:   NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %normal_ret_a0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %noalias_ret_a0, double* %nocap_callee_arg
; IPAA:   NoAlias:	double* %arg_a0, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %arg_a1, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %noalias_ret_a0, double** %indirect_a0
; IPAA:   NoAlias:	double* %noalias_ret_a0, double** %indirect_a1
; IPAA:   NoAlias:	double* %loaded_a0, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %loaded_a1, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a0
; IPAA:   NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a0
; IPAA:   NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_ret_a1
; IPAA:   MayAlias:	double* %noalias_ret_a1, double* %nocap_callee_arg
; IPAA:   NoAlias:	double* %arg_a0, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %arg_a1, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %noalias_ret_a1, double** %indirect_a0
; IPAA:   NoAlias:	double* %noalias_ret_a1, double** %indirect_a1
; IPAA:   NoAlias:	double* %loaded_a0, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %loaded_a1, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a1
; IPAA:   NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a0
; IPAA:   NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a1
; IPAA:   NoAlias:	double* %noalias_ret_a0, double* %noalias_ret_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %callee_arg
; IPAA:   MayAlias:	double* %arg_b0, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %arg_b0
; IPAA:   MayAlias:	double* %arg_a1, double* %arg_b0
; IPAA:   MayAlias:	double* %arg_b0, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %arg_b0, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %arg_b0, double** %indirect_a0
; IPAA:   MayAlias:	double* %arg_b0, double** %indirect_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %loaded_a0
; IPAA:   MayAlias:	double* %arg_b0, double* %loaded_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %escape_alloca_a0
; IPAA:   MayAlias:	double* %arg_b0, double* %escape_alloca_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %arg_b0, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %arg_b0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %arg_b0, double* %noalias_ret_a1
; IPAA:   MayAlias:	double* %arg_b1, double* %callee_arg
; IPAA:   MayAlias:	double* %arg_b1, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %arg_b1
; IPAA:   MayAlias:	double* %arg_a1, double* %arg_b1
; IPAA:   MayAlias:	double* %arg_b1, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %arg_b1, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %arg_b1, double** %indirect_a0
; IPAA:   MayAlias:	double* %arg_b1, double** %indirect_a1
; IPAA:   MayAlias:	double* %arg_b1, double* %loaded_a0
; IPAA:   MayAlias:	double* %arg_b1, double* %loaded_a1
; IPAA:   MayAlias:	double* %arg_b1, double* %escape_alloca_a0
; IPAA:   MayAlias:	double* %arg_b1, double* %escape_alloca_a1
; IPAA:   MayAlias:	double* %arg_b1, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %arg_b1, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %arg_b1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %arg_b1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %arg_b1, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %arg_b1, double* %noalias_ret_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %arg_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %noalias_arg_b0, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %arg_a1, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %noalias_arg_b0, double** %indirect_a0
; IPAA:   MayAlias:	double* %noalias_arg_b0, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %loaded_a1, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %noalias_arg_b0, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %noalias_arg_b0, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %noalias_arg_b0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %noalias_arg_b0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %noalias_arg_b0, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %noalias_arg_b0, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %arg_b0, double* %noalias_arg_b0
; IPAA:   NoAlias:	double* %arg_b1, double* %noalias_arg_b0
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %noalias_arg_b1, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %arg_a1, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %noalias_arg_b1, double** %indirect_a0
; IPAA:   MayAlias:	double* %noalias_arg_b1, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %loaded_a1, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %noalias_arg_b1, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %noalias_arg_b1, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %noalias_arg_b1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %noalias_arg_b1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %noalias_arg_b1, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %noalias_arg_b1, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %arg_b0, double* %noalias_arg_b1
; IPAA:   NoAlias:	double* %arg_b1, double* %noalias_arg_b1
; IPAA:   NoAlias:	double* %noalias_arg_b0, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %callee_arg, double** %indirect_b0
; IPAA:   MayAlias:	double* %nocap_callee_arg, double** %indirect_b0
; IPAA:   MayAlias:	double* %arg_a0, double** %indirect_b0
; IPAA:   MayAlias:	double* %arg_a1, double** %indirect_b0
; IPAA:   MayAlias:	double* %noalias_arg_a0, double** %indirect_b0
; IPAA:   MayAlias:	double* %noalias_arg_a1, double** %indirect_b0
; IPAA:   MayAlias:	double** %indirect_a0, double** %indirect_b0
; IPAA:   MayAlias:	double** %indirect_a1, double** %indirect_b0
; IPAA:   MayAlias:	double* %loaded_a0, double** %indirect_b0
; IPAA:   MayAlias:	double* %loaded_a1, double** %indirect_b0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double** %indirect_b0
; IPAA:   MayAlias:	double* %escape_alloca_a1, double** %indirect_b0
; IPAA:   MayAlias:	double* %noescape_alloca_a0, double** %indirect_b0
; IPAA:   MayAlias:	double* %noescape_alloca_a1, double** %indirect_b0
; IPAA:   MayAlias:	double* %normal_ret_a0, double** %indirect_b0
; IPAA:   MayAlias:	double* %normal_ret_a1, double** %indirect_b0
; IPAA:   MayAlias:	double* %noalias_ret_a0, double** %indirect_b0
; IPAA:   MayAlias:	double* %noalias_ret_a1, double** %indirect_b0
; IPAA:   MayAlias:	double* %arg_b0, double** %indirect_b0
; IPAA:   MayAlias:	double* %arg_b1, double** %indirect_b0
; IPAA:   NoAlias:	double* %noalias_arg_b0, double** %indirect_b0
; IPAA:   NoAlias:	double* %noalias_arg_b1, double** %indirect_b0
; IPAA:   MayAlias:	double* %callee_arg, double** %indirect_b1
; IPAA:   MayAlias:	double* %nocap_callee_arg, double** %indirect_b1
; IPAA:   MayAlias:	double* %arg_a0, double** %indirect_b1
; IPAA:   MayAlias:	double* %arg_a1, double** %indirect_b1
; IPAA:   MayAlias:	double* %noalias_arg_a0, double** %indirect_b1
; IPAA:   MayAlias:	double* %noalias_arg_a1, double** %indirect_b1
; IPAA:   MayAlias:	double** %indirect_a0, double** %indirect_b1
; IPAA:   MayAlias:	double** %indirect_a1, double** %indirect_b1
; IPAA:   MayAlias:	double* %loaded_a0, double** %indirect_b1
; IPAA:   MayAlias:	double* %loaded_a1, double** %indirect_b1
; IPAA:   MayAlias:	double* %escape_alloca_a0, double** %indirect_b1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double** %indirect_b1
; IPAA:   MayAlias:	double* %noescape_alloca_a0, double** %indirect_b1
; IPAA:   MayAlias:	double* %noescape_alloca_a1, double** %indirect_b1
; IPAA:   MayAlias:	double* %normal_ret_a0, double** %indirect_b1
; IPAA:   MayAlias:	double* %normal_ret_a1, double** %indirect_b1
; IPAA:   MayAlias:	double* %noalias_ret_a0, double** %indirect_b1
; IPAA:   MayAlias:	double* %noalias_ret_a1, double** %indirect_b1
; IPAA:   MayAlias:	double* %arg_b0, double** %indirect_b1
; IPAA:   MayAlias:	double* %arg_b1, double** %indirect_b1
; IPAA:   NoAlias:	double* %noalias_arg_b0, double** %indirect_b1
; IPAA:   NoAlias:	double* %noalias_arg_b1, double** %indirect_b1
; IPAA:   MayAlias:	double** %indirect_b0, double** %indirect_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %loaded_b0
; IPAA:   MayAlias:	double* %loaded_b0, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %loaded_b0
; IPAA:   MayAlias:	double* %arg_a1, double* %loaded_b0
; IPAA:   MayAlias:	double* %loaded_b0, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %loaded_b0, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %loaded_b0, double** %indirect_a0
; IPAA:   MayAlias:	double* %loaded_b0, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %loaded_b0
; IPAA:   MayAlias:	double* %loaded_a1, double* %loaded_b0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %loaded_b0
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %loaded_b0
; IPAA:   MayAlias:	double* %loaded_b0, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %loaded_b0, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %loaded_b0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %loaded_b0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %loaded_b0, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %loaded_b0, double* %noalias_ret_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %loaded_b0
; IPAA:   MayAlias:	double* %arg_b1, double* %loaded_b0
; IPAA:   NoAlias:	double* %loaded_b0, double* %noalias_arg_b0
; IPAA:   NoAlias:	double* %loaded_b0, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %loaded_b0, double** %indirect_b0
; IPAA:   MayAlias:	double* %loaded_b0, double** %indirect_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %loaded_b1
; IPAA:   MayAlias:	double* %loaded_b1, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %loaded_b1
; IPAA:   MayAlias:	double* %arg_a1, double* %loaded_b1
; IPAA:   MayAlias:	double* %loaded_b1, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %loaded_b1, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %loaded_b1, double** %indirect_a0
; IPAA:   MayAlias:	double* %loaded_b1, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %loaded_b1
; IPAA:   MayAlias:	double* %loaded_a1, double* %loaded_b1
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %loaded_b1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %loaded_b1
; IPAA:   MayAlias:	double* %loaded_b1, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %loaded_b1, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %loaded_b1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %loaded_b1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %loaded_b1, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %loaded_b1, double* %noalias_ret_a1
; IPAA:   MayAlias:	double* %arg_b0, double* %loaded_b1
; IPAA:   MayAlias:	double* %arg_b1, double* %loaded_b1
; IPAA:   NoAlias:	double* %loaded_b1, double* %noalias_arg_b0
; IPAA:   NoAlias:	double* %loaded_b1, double* %noalias_arg_b1
; IPAA:   MayAlias:	double* %loaded_b1, double** %indirect_b0
; IPAA:   MayAlias:	double* %loaded_b1, double** %indirect_b1
; IPAA:   MayAlias:	double* %loaded_b0, double* %loaded_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %escape_alloca_b0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %escape_alloca_b0
; IPAA:   MayAlias:	double* %arg_a1, double* %escape_alloca_b0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %escape_alloca_b0, double** %indirect_a0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double** %indirect_a1
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %loaded_a0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %loaded_a1
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %escape_alloca_b0
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %escape_alloca_b0
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %arg_b0, double* %escape_alloca_b0
; IPAA:   NoAlias:	double* %arg_b1, double* %escape_alloca_b0
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_arg_b0
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_arg_b1
; IPAA:   NoAlias:	double* %escape_alloca_b0, double** %indirect_b0
; IPAA:   NoAlias:	double* %escape_alloca_b0, double** %indirect_b1
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %loaded_b0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %loaded_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %escape_alloca_b1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %escape_alloca_b1
; IPAA:   MayAlias:	double* %arg_a1, double* %escape_alloca_b1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %noalias_arg_a0
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %noalias_arg_a1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double** %indirect_a0
; IPAA:   MayAlias:	double* %escape_alloca_b1, double** %indirect_a1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %loaded_a0
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %loaded_a1
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %escape_alloca_b1
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %escape_alloca_b1
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noescape_alloca_a0
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %noalias_ret_a0
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %noalias_ret_a1
; IPAA:   NoAlias:	double* %arg_b0, double* %escape_alloca_b1
; IPAA:   NoAlias:	double* %arg_b1, double* %escape_alloca_b1
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_arg_b0
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_arg_b1
; IPAA:   NoAlias:	double* %escape_alloca_b1, double** %indirect_b0
; IPAA:   NoAlias:	double* %escape_alloca_b1, double** %indirect_b1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %loaded_b0
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %loaded_b1
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %escape_alloca_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %arg_a0, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %arg_a1, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %noescape_alloca_b0, double** %indirect_a0
; IPAA:   MayAlias:	double* %noescape_alloca_b0, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %loaded_a1, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %noescape_alloca_a0, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %noescape_alloca_a1, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %noescape_alloca_b0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %noescape_alloca_b0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %noalias_ret_a0, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %noalias_ret_a1, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %arg_b0, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %arg_b1, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %noalias_arg_b0, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %noalias_arg_b1, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %noescape_alloca_b0, double** %indirect_b0
; IPAA:   NoAlias:	double* %noescape_alloca_b0, double** %indirect_b1
; IPAA:   NoAlias:	double* %loaded_b0, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %loaded_b1, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noescape_alloca_b0
; IPAA:   MayAlias:	double* %callee_arg, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %arg_a0, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %arg_a1, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %noescape_alloca_b1, double** %indirect_a0
; IPAA:   MayAlias:	double* %noescape_alloca_b1, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %loaded_a1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noescape_alloca_a0, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noescape_alloca_a1, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %noescape_alloca_b1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %noescape_alloca_b1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %noalias_ret_a0, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %noalias_ret_a1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %arg_b0, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %arg_b1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noalias_arg_b0, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noalias_arg_b1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noescape_alloca_b1, double** %indirect_b0
; IPAA:   NoAlias:	double* %noescape_alloca_b1, double** %indirect_b1
; IPAA:   NoAlias:	double* %loaded_b0, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %loaded_b1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noescape_alloca_b0, double* %noescape_alloca_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %arg_a0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %arg_a1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %normal_ret_b0, double** %indirect_a0
; IPAA:   MayAlias:	double* %normal_ret_b0, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %loaded_a1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %noescape_alloca_a0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %noescape_alloca_a1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %normal_ret_a0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %normal_ret_a1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %noalias_ret_a0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %noalias_ret_a1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %arg_b0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %arg_b1, double* %normal_ret_b0
; IPAA:   NoAlias:	double* %noalias_arg_b0, double* %normal_ret_b0
; IPAA:   NoAlias:	double* %noalias_arg_b1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %normal_ret_b0, double** %indirect_b0
; IPAA:   MayAlias:	double* %normal_ret_b0, double** %indirect_b1
; IPAA:   MayAlias:	double* %loaded_b0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %loaded_b1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %normal_ret_b0
; IPAA:   NoAlias:	double* %noescape_alloca_b0, double* %normal_ret_b0
; IPAA:   NoAlias:	double* %noescape_alloca_b1, double* %normal_ret_b0
; IPAA:   MayAlias:	double* %callee_arg, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %nocap_callee_arg, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %arg_a0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %arg_a1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %normal_ret_b1, double** %indirect_a0
; IPAA:   MayAlias:	double* %normal_ret_b1, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %loaded_a1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %noescape_alloca_a0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %noescape_alloca_a1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %normal_ret_a0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %normal_ret_a1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %noalias_ret_a0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %noalias_ret_a1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %arg_b0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %arg_b1, double* %normal_ret_b1
; IPAA:   NoAlias:	double* %noalias_arg_b0, double* %normal_ret_b1
; IPAA:   NoAlias:	double* %noalias_arg_b1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %normal_ret_b1, double** %indirect_b0
; IPAA:   MayAlias:	double* %normal_ret_b1, double** %indirect_b1
; IPAA:   MayAlias:	double* %loaded_b0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %loaded_b1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %escape_alloca_b0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %escape_alloca_b1, double* %normal_ret_b1
; IPAA:   NoAlias:	double* %noescape_alloca_b0, double* %normal_ret_b1
; IPAA:   NoAlias:	double* %noescape_alloca_b1, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %normal_ret_b0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %noalias_ret_b0, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %arg_a1, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %noalias_ret_b0, double** %indirect_a0
; IPAA:   MayAlias:	double* %noalias_ret_b0, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %loaded_a1, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %noalias_ret_b0, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %noalias_ret_b0, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %noalias_ret_b0, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %noalias_ret_b0, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %noalias_ret_a0, double* %noalias_ret_b0
; IPAA:   MayAlias:	double* %noalias_ret_a1, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %arg_b0, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %arg_b1, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %noalias_arg_b0, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %noalias_arg_b1, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %noalias_ret_b0, double** %indirect_b0
; IPAA:   NoAlias:	double* %noalias_ret_b0, double** %indirect_b1
; IPAA:   NoAlias:	double* %loaded_b0, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %loaded_b1, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_ret_b0
; IPAA:   NoAlias:	double* %noalias_ret_b0, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %noalias_ret_b0, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noalias_ret_b0, double* %normal_ret_b0
; IPAA:   NoAlias:	double* %noalias_ret_b0, double* %normal_ret_b1
; IPAA:   MayAlias:	double* %callee_arg, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %noalias_ret_b1, double* %nocap_callee_arg
; IPAA:   MayAlias:	double* %arg_a0, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %arg_a1, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %noalias_arg_a0, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %noalias_arg_a1, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %noalias_ret_b1, double** %indirect_a0
; IPAA:   MayAlias:	double* %noalias_ret_b1, double** %indirect_a1
; IPAA:   MayAlias:	double* %loaded_a0, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %loaded_a1, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %escape_alloca_a0, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %escape_alloca_a1, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %noalias_ret_b1, double* %noescape_alloca_a0
; IPAA:   MayAlias:	double* %noalias_ret_b1, double* %noescape_alloca_a1
; IPAA:   MayAlias:	double* %noalias_ret_b1, double* %normal_ret_a0
; IPAA:   MayAlias:	double* %noalias_ret_b1, double* %normal_ret_a1
; IPAA:   MayAlias:	double* %noalias_ret_a0, double* %noalias_ret_b1
; IPAA:   MayAlias:	double* %noalias_ret_a1, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %arg_b0, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %arg_b1, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %noalias_arg_b0, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %noalias_arg_b1, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %noalias_ret_b1, double** %indirect_b0
; IPAA:   NoAlias:	double* %noalias_ret_b1, double** %indirect_b1
; IPAA:   NoAlias:	double* %loaded_b0, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %loaded_b1, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %escape_alloca_b0, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %escape_alloca_b1, double* %noalias_ret_b1
; IPAA:   NoAlias:	double* %noalias_ret_b1, double* %noescape_alloca_b0
; IPAA:   NoAlias:	double* %noalias_ret_b1, double* %noescape_alloca_b1
; IPAA:   NoAlias:	double* %noalias_ret_b1, double* %normal_ret_b0
; IPAA:   NoAlias:	double* %noalias_ret_b1, double* %normal_ret_b1
; IPAA:   NoAlias:	double* %noalias_ret_b0, double* %noalias_ret_b1
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  %normal_ret_a0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  %normal_ret_a1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  %noalias_ret_a0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  %noalias_ret_a1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  call void @callee(double* %escape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  call void @callee(double* %escape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_a0)
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_a1)
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  %normal_ret_b0 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  %normal_ret_b1 = call double* @normal_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b0	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  %noalias_ret_b0 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %noalias_ret_b1	<->  %noalias_ret_b1 = call double* @noalias_returner() ; <double*> [#uses=1]
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @callee(double* %escape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @callee(double* %escape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b0)
; IPAA:     ModRef:  Ptr: double* %callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %nocap_callee_arg	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_arg_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double** %indirect_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %loaded_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %escape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noescape_alloca_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %noescape_alloca_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:     ModRef:  Ptr: double* %normal_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b0	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA:   NoModRef:  Ptr: double* %noalias_ret_b1	<->  call void @nocap_callee(double* %noescape_alloca_b1)
; IPAA: ===== Alias Analysis Evaluator Report =====
; IPAA:   561 Total Alias Queries Performed
; IPAA:   184 no alias responses (32.7%)
; IPAA:   377 may alias responses (67.2%)
; IPAA:   0 must alias responses (0.0%)
; IPAA:   Alias Analysis Evaluator Pointer Alias Summary: 32%/67%/0%
; IPAA:   544 Total ModRef Queries Performed
; IPAA:   88 no mod/ref responses (16.1%)
; IPAA:   0 mod responses (0.0%)
; IPAA:   0 ref responses (0.0%)
; IPAA:   456 mod & ref responses (83.8%)
; IPAA:   Alias Analysis Evaluator Mod/Ref Summary: 16%/0%/0%/83%
