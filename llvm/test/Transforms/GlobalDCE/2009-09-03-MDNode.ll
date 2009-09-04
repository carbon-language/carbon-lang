; RUN: llvm-as < %s | opt -globaldce | llc -O0 -o /dev/null

%struct..0__pthread_mutex_s = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_list_t }
%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>" = type { i32 }
%struct.__pthread_list_t = type { %struct.__pthread_list_t*, %struct.__pthread_list_t* }
%struct.pthread_attr_t = type { i64, [48 x i8] }
%struct.pthread_mutex_t = type { %struct..0__pthread_mutex_s }

@_ZL20__gthrw_pthread_oncePiPFvvE = alias weak i32 (i32*, void ()*)* @pthread_once ; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = alias weak i8* (i32)* @pthread_getspecific ; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = alias weak i32 (i32, i8*)* @pthread_setspecific ; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = alias weak i32 (i64*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create ; <i32 (i64*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = alias weak i32 (i64)* @pthread_cancel ; <i32 (i64)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock ; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock ; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock ; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = alias weak i32 (%struct.pthread_mutex_t*, %"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)* @pthread_mutex_init ; <i32 (%struct.pthread_mutex_t*, %"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create ; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = alias weak i32 (i32)* @pthread_key_delete ; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = alias weak i32 (%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)* @pthread_mutexattr_init ; <i32 (%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = alias weak i32 (%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*, i32)* @pthread_mutexattr_settype ; <i32 (%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = alias weak i32 (%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)* @pthread_mutexattr_destroy ; <i32 (%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)*> [#uses=0]

define weak void @_ZN9__gnu_cxx26__aux_require_boolean_exprIbEEvRKT_(i8* %__t) {
entry:
  tail call void @llvm.dbg.func.start(metadata !0)
  tail call void @llvm.dbg.stoppoint(i32 240, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !0)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_19_ConvertibleConceptIjjEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !8)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !8)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPcEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !11)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !11)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPKcEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !12)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !12)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPwEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !13)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !13)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPKwEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !14)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !14)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIPwEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !15)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !15)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIPcEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !16)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !16)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIiEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !17)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !17)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIlEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !18)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !18)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIxEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !19)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !19)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIjEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !20)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !20)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_22_OutputIteratorConceptISt19ostreambuf_iteratorIcSt11char_traitsIcEEcEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !21)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !21)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_22_OutputIteratorConceptISt19ostreambuf_iteratorIwSt11char_traitsIwEEwEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !22)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !22)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPcEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !23)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !23)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPKcEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !24)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !24)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPKcSsEEEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !25)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !25)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPcSsEEEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !26)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !26)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPKwSbIwSt11char_traitsIwESaIwEEEEEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !27)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !27)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPwSbIwSt11char_traitsIwESaIwEEEEEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !28)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !28)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPwEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !29)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !29)
  ret void
}

define weak void @_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPKwEEEEvv() {
entry:
  tail call void @llvm.dbg.func.start(metadata !30)
  tail call void @llvm.dbg.stoppoint(i32 63, i32 0, metadata !2)
  tail call void @llvm.dbg.region.end(metadata !30)
  ret void
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

declare extern_weak i32 @pthread_once(i32*, void ()*)

declare extern_weak i8* @pthread_getspecific(i32)

declare extern_weak i32 @pthread_setspecific(i32, i8*)

declare extern_weak i32 @pthread_create(i64*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)

declare extern_weak i32 @pthread_cancel(i64)

declare extern_weak i32 @pthread_mutex_lock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_trylock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_unlock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_init(%struct.pthread_mutex_t*, %"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)

declare extern_weak i32 @pthread_key_create(i32*, void (i8*)*)

declare extern_weak i32 @pthread_key_delete(i32)

declare extern_weak i32 @pthread_mutexattr_init(%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)

declare extern_weak i32 @pthread_mutexattr_settype(%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*, i32)

declare extern_weak i32 @pthread_mutexattr_destroy(%"struct.__gnu_cxx::_ConvertibleConcept<unsigned int,unsigned int>"*)

!0 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__aux_require_boolean_expr<bool>", metadata !"__aux_require_boolean_expr<bool>", metadata !"_ZN9__gnu_cxx26__aux_require_boolean_exprIbEEvRKT_", metadata !2, i32 239, metadata !3, i1 false, i1 true}
!1 = metadata !{i32 458769, i32 0, i32 4, metadata !"concept-inst.cc", metadata !"/home/buildbot/buildslave/llvm-x86_64-linux-selfhost/llvm-gcc.obj/x86_64-unknown-linux-gnu/libstdc++-v3/src/../../../../llvm-gcc.src/libstdc++-v3/src", metadata !"4.2.1 (Based on Apple Inc. build 5649) (LLVM build)", i1 true, i1 true, metadata !"", i32 0}
!2 = metadata !{i32 458769, i32 0, i32 4, metadata !"boost_concept_check.h", metadata !"/home/buildbot/buildslave/llvm-x86_64-linux-selfhost/llvm-gcc.obj/x86_64-unknown-linux-gnu/libstdc++-v3/include/bits", metadata !"4.2.1 (Based on Apple Inc. build 5649) (LLVM build)", i1 false, i1 true, metadata !"", i32 0}
!3 = metadata !{i32 458773, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0}
!4 = metadata !{null, metadata !5}
!5 = metadata !{i32 458768, metadata !1, metadata !"", metadata !1, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !6}
!6 = metadata !{i32 458790, metadata !1, metadata !"", metadata !1, i32 0, i64 8, i64 8, i64 0, i32 0, metadata !7}
!7 = metadata !{i32 458788, metadata !1, metadata !"bool", metadata !1, i32 0, i64 8, i64 8, i64 0, i32 0, i32 2}
!8 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_ConvertibleConcept<unsigned int, unsigned int> >", metadata !"__function_requires<__gnu_cxx::_ConvertibleConcept<unsigned int, unsigned int> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_19_ConvertibleConceptIjjEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!9 = metadata !{i32 458773, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0}
!10 = metadata !{null}
!11 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<char*> >", metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<char*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPcEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!12 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<const char*> >", metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<const char*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPKcEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!13 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<wchar_t*> >", metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<wchar_t*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPwEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!14 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<const wchar_t*> >", metadata !"__function_requires<__gnu_cxx::_InputIteratorConcept<const wchar_t*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_21_InputIteratorConceptIPKwEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!15 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<wchar_t*> >", metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<wchar_t*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIPwEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!16 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<char*> >", metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<char*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIPcEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!17 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<int> >", metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<int> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIiEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!18 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<long int> >", metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<long int> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIlEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!19 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<long long int> >", metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<long long int> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIxEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!20 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<unsigned int> >", metadata !"__function_requires<__gnu_cxx::_LessThanComparableConcept<unsigned int> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_26_LessThanComparableConceptIjEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!21 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_OutputIteratorConcept<std::ostreambuf_iterator<char, std::char_traits<char> >, char> >", metadata !"__function_requires<__gnu_cxx::_OutputIteratorConcept<std::ostreambuf_iterator<char, std::char_traits<char> >, char> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_22_OutputIteratorConceptISt19ostreambuf_iteratorIcSt11char_traitsIcEEcEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!22 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_OutputIteratorConcept<std::ostreambuf_iterator<wchar_t, std::char_traits<wchar_t> >, wchar_t> >", metadata !"__function_requires<__gnu_cxx::_OutputIteratorConcept<std::ostreambuf_iterator<wchar_t, std::char_traits<wchar_t> >, wchar_t> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_22_OutputIteratorConceptISt19ostreambuf_iteratorIwSt11char_traitsIwEEwEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!23 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<char*> >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<char*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPcEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!24 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<const char*> >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<const char*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPKcEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!25 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<const char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<const char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPKcSsEEEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!26 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<char*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPcSsEEEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!27 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<const wchar_t*, std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > > > >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<const wchar_t*, std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > > > >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPKwSbIwSt11char_traitsIwESaIwEEEEEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!28 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<wchar_t*, std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > > > >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<__gnu_cxx::__normal_iterator<wchar_t*, std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > > > >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptINS_17__normal_iteratorIPwSbIwSt11char_traitsIwESaIwEEEEEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!29 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<wchar_t*> >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<wchar_t*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPwEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
!30 = metadata !{i32 458798, i32 0, metadata !1, metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<const wchar_t*> >", metadata !"__function_requires<__gnu_cxx::_RandomAccessIteratorConcept<const wchar_t*> >", metadata !"_ZN9__gnu_cxx19__function_requiresINS_28_RandomAccessIteratorConceptIPKwEEEEvv", metadata !2, i32 61, metadata !9, i1 false, i1 true}
