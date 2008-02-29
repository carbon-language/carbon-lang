; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin -regalloc=local
; PR1925

	%struct..0._10 = type { i32 }
	%struct..1__pthread_mutex_s = type { i32, i32, i32, i32, i32, %struct..0._10 }
	%"struct.kc::impl_Ccode_option" = type { %"struct.kc::impl_abstract_phylum" }
	%"struct.kc::impl_ID" = type { %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_Ccode_option"*, %"struct.kc::impl_casestring__Str"*, i32, %"struct.kc::impl_casestring__Str"* }
	%"struct.kc::impl_abstract_phylum" = type { i32 (...)** }
	%"struct.kc::impl_casestring__Str" = type { %"struct.kc::impl_abstract_phylum", i8* }
	%struct.pthread_attr_t = type { i32, [32 x i8] }
	%struct.pthread_mutex_t = type { %struct..1__pthread_mutex_s }

@_ZL20__gthrw_pthread_oncePiPFvvE = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE3 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj4 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv5 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_6 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm7 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t8 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t9 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t10 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t11 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE12 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej13 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t14 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti15 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t16 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE19 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj20 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv21 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_22 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm23 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t24 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t25 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t26 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t27 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE28 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej29 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t30 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti31 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t32 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE45 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj46 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv47 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_48 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm49 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t50 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t51 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t52 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t53 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE54 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej55 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t56 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti57 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t58 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE113 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj114 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv115 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_116 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm117 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t118 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t119 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t120 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t121 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE122 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej123 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t124 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti125 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t126 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE547 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj548 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv549 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_550 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm551 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t552 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t553 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t554 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t555 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE556 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej557 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t558 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti559 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t560 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE820 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj821 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv822 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_823 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm824 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t825 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t826 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t827 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t828 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE829 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej830 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t831 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti832 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t833 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE963 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj964 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv965 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_966 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm967 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t968 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t969 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t970 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t971 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE972 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej973 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t974 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti975 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t976 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE1058 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj1059 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv1060 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_1061 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm1062 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t1063 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t1064 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t1065 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t1066 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE1067 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej1068 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t1069 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti1070 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t1071 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE1123 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj1124 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv1125 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_1126 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm1127 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t1128 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t1129 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t1130 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t1131 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE1132 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej1133 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t1134 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti1135 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t1136 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE1179 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj1180 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv1181 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_1182 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm1183 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t1184 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t1185 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t1186 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t1187 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE1188 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej1189 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t1190 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti1191 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t1192 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE1195 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj1196 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv1197 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_1198 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm1199 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t1200 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t1201 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t1202 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t1203 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE1204 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej1205 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t1206 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti1207 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t1208 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE1749 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj1750 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv1751 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_1752 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm1753 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t1754 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t1755 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t1756 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t1757 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE1758 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej1759 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t1760 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti1761 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t1762 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL20__gthrw_pthread_oncePiPFvvE1817 = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj1818 = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv1819 = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_1820 = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm1821 = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t1822 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t1823 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t1824 = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t1825 = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._10*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._10*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE1826 = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej1827 = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t1828 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_init		; <i32 (%struct..0._10*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti1829 = alias weak i32 (%struct..0._10*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._10*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t1830 = alias weak i32 (%struct..0._10*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._10*)*> [#uses=0]

declare i32 @pthread_once(i32*, void ()*)

declare i8* @pthread_getspecific(i32)

declare i32 @pthread_setspecific(i32, i8*)

declare i32 @pthread_create(i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)

declare i32 @pthread_cancel(i32)

declare i32 @pthread_mutex_lock(%struct.pthread_mutex_t*)

declare i32 @pthread_mutex_trylock(%struct.pthread_mutex_t*)

declare i32 @pthread_mutex_unlock(%struct.pthread_mutex_t*)

declare i32 @pthread_mutex_init(%struct.pthread_mutex_t*, %struct..0._10*)

declare i32 @pthread_key_create(i32*, void (i8*)*)

declare i32 @pthread_key_delete(i32)

declare i32 @pthread_mutexattr_init(%struct..0._10*)

declare i32 @pthread_mutexattr_settype(%struct..0._10*, i32)

declare i32 @pthread_mutexattr_destroy(%struct..0._10*)

define %"struct.kc::impl_ID"* @_ZN2kc18f_typeofunpsubtermEPNS_15impl_unpsubtermEPNS_7impl_IDE(%"struct.kc::impl_Ccode_option"* %a_unpsubterm, %"struct.kc::impl_ID"* %a_operator) {
entry:
	%tmp8 = getelementptr %"struct.kc::impl_Ccode_option"* %a_unpsubterm, i32 0, i32 0, i32 0		; <i32 (...)***> [#uses=0]
	br i1 false, label %bb41, label %bb55

bb41:		; preds = %entry
	ret %"struct.kc::impl_ID"* null

bb55:		; preds = %entry
	%tmp67 = tail call i32 null( %"struct.kc::impl_abstract_phylum"* null )		; <i32> [#uses=0]
	%tmp97 = tail call i32 null( %"struct.kc::impl_abstract_phylum"* null )		; <i32> [#uses=0]
	ret %"struct.kc::impl_ID"* null
}
