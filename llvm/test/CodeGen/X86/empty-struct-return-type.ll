; RUN: llvm-as < %s | llc -march=x86-64 | grep call | count 3
; PR4688

; Return types can be empty structs, which can be awkward.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%llvm.fe.ty10 = type %struct.pthread_attr_t*
	%llvm.fe.ty11 = type i8* (i8*)
	%llvm.fe.ty12 = type %llvm.fe.ty11*
	%llvm.fe.ty13 = type %struct.__pthread_list_t*
	%llvm.fe.ty15 = type %struct.pthread_mutex_t*
	%llvm.fe.ty17 = type %struct.QBasicAtomicInt*
	%llvm.fe.ty18 = type void (i8*)
	%llvm.fe.ty19 = type %llvm.fe.ty18*
	%llvm.fe.ty3 = type void ()
	%llvm.fe.ty4 = type %llvm.fe.ty3*
	%struct..0KnownPointers = type { %struct.QMutex, %"struct.QHash<void*,QHashDummyValue>" }
	%struct..0__pthread_mutex_s = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_list_t }
	%struct.QBasicAtomicInt = type { i32 }
	%"struct.QHash<void*,QHashDummyValue>" = type { %"struct.QHash<void*,QHashDummyValue>::._177" }
	%"struct.QHash<void*,QHashDummyValue>::._177" = type { %struct.QHashData* }
	%struct.QHashData = type { %"struct.QHashData::Node"*, %"struct.QHashData::Node"**, %struct.QBasicAtomicInt, i32, i32, i16, i16, i32, i8 }
	%"struct.QHashData::Node" = type { %"struct.QHashData::Node"*, i32 }
	%struct.QMutex = type { %struct.QMutexPrivate* }
	%struct.QMutexPrivate = type opaque
	%struct.__pthread_list_t = type { %llvm.fe.ty13, %llvm.fe.ty13 }
	%struct.pthread_attr_t = type { i64, [48 x i8] }
	%struct.pthread_mutex_t = type { %struct..0__pthread_mutex_s }

define void @_ZN15QtSharedPointer22internalSafetyCheckAddEPVKv(i8* %ptr) {
entry:
	%0 = invoke fastcc %struct..0KnownPointers* @_ZL13knownPointersv()
			to label %invcont1 unwind label %lpad		; <%struct..0KnownPointers*> [#uses=0]

invcont1:		; preds = %entry
	%1 = invoke fastcc %struct..0KnownPointers* @_ZL13knownPointersv()
			to label %invcont3 unwind label %lpad		; <%struct..0KnownPointers*> [#uses=0]

invcont3:		; preds = %invcont1
	%2 = call { } @_ZNK5QHashIPv15QHashDummyValueE5valueERKS0_(%"struct.QHash<void*,QHashDummyValue>"* undef, i8** undef)		; <{ }> [#uses=0]
	unreachable

lpad:		; preds = %invcont1, %entry
	unreachable
}

declare hidden { } @_ZNK5QHashIPv15QHashDummyValueE5valueERKS0_(%"struct.QHash<void*,QHashDummyValue>"*, i8** nocapture) nounwind

declare fastcc %struct..0KnownPointers* @_ZL13knownPointersv()
