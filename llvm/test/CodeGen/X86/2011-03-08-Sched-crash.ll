; RUN: llc < %s
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.0.0"

%0 = type { %"class.JSC::ExecutablePool"**, i32, i32, %2 }
%1 = type { %"class.js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry"*, i32, i32, %2 }
%2 = type { %"union.js::AlignedStorage<16>::U" }
%3 = type { i16* }
%4 = type { %5 }
%5 = type { %6, %6 }
%6 = type { %struct.JSString* }
%7 = type { %"struct.js::gc::ArenaHeader" }
%8 = type { double }
%9 = type { %7, %10 }
%10 = type { [4072 x %"union.js::gc::ThingOrCell"] }
%11 = type { %"struct.js::Shape"* }
%12 = type { i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)* }
%13 = type { i32 (%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)* }
%14 = type { %"class.js::KidsPointer" }
%15 = type { %struct.JSObject* }
%16 = type { %"class.JSC::ExecutablePool"**, i32, i32, %"class.js::Value" }
%17 = type { %"struct.js::VMSideExit"**, i32, i32, %"class.nanojit::Allocator"* }
%18 = type { %"class.js::Value"*, i32, i32, %"class.nanojit::Allocator"* }
%19 = type { %"struct.js::Shape"**, i32, i32, %"class.nanojit::Allocator"* }
%20 = type { %"class.nanojit::CodeList"* }
%21 = type { %"class.nanojit::Allocator"*, i32, %22** }
%22 = type opaque
%23 = type { %"class.nanojit::Allocator"*, i32, %24** }
%24 = type opaque
%25 = type { %"class.nanojit::Allocator"*, i32, %26** }
%26 = type opaque
%27 = type { %"class.nanojit::Allocator"*, i32, %28** }
%28 = type opaque
%29 = type { %30 }
%30 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::FrameInfo *const, js::HashSet<js::FrameInfo *, js::FrameInfoCache::HashPolicy, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%31 = type { i8**, i32, i32, %"class.nanojit::Allocator"* }
%32 = type { %33 }
%33 = type { [4 x i8], i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry, js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::MapHashPolicy, js::ContextAllocPolicy>::Entry"* }
%34 = type { [4 x i8], i32*, i32, i32, %"class.js::Value" }
%35 = type { [4 x i8], i8*, i32, i32, %36 }
%36 = type { %"union.js::AlignedStorage<256>::U" }
%37 = type { %struct.JSScript* }
%38 = type { i16*, i32, i32, %"class.nanojit::Allocator"* }
%39 = type { %40 }
%40 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%41 = type { %42 }
%42 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%43 = type { %44 }
%44 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%45 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<js::Value, js::Value, js::WrapperHasher, js::SystemAllocPolicy>::Entry, js::HashMap<js::Value, js::Value, js::WrapperHasher, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%46 = type { i8*, i8*, i8* }
%47 = type { %48 }
%48 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::EmptyShape *const, js::HashSet<js::EmptyShape *, js::DefaultHasher<js::EmptyShape *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%49 = type { %50 }
%50 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<JSFunction *, JSString *, js::DefaultHasher<JSFunction *>, js::SystemAllocPolicy>::Entry, js::HashMap<JSFunction *, JSString *, js::DefaultHasher<JSFunction *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%51 = type { %struct.JSCompartment**, i32, i32, %"class.js::Value" }
%52 = type { %53 }
%53 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::gc::Chunk *const, js::HashSet<js::gc::Chunk *, js::GCChunkHasher, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%54 = type { %55 }
%55 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<void *, js::RootInfo, js::DefaultHasher<void *>, js::SystemAllocPolicy>::Entry, js::HashMap<void *, js::RootInfo, js::DefaultHasher<void *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%56 = type { %57 }
%57 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<void *, unsigned int, js::GCPtrHasher, js::SystemAllocPolicy>::Entry, js::HashMap<void *, unsigned int, js::GCPtrHasher, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%58 = type { i32 (%struct.JSContext*, i32, %"class.js::Value"*)*, %"struct.js::Class"*, %struct.JSNativeTraceInfo* }
%59 = type { [18 x i32] }
%60 = type { i8***, i32, i32, %61 }
%61 = type { %"union.js::AlignedStorage<64>::U" }
%62 = type { %63 }
%63 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<js::HashMap<void *, JSThread *, js::DefaultHasher<void *>, js::SystemAllocPolicy>::Entry, js::HashMap<void *, JSThread *, js::DefaultHasher<void *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%64 = type { %65 }
%65 = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<const unsigned long, js::HashSet<unsigned long, js::AtomHasher, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%66 = type { %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom* }
%67 = type { %"struct.js::SlotMap::SlotInfo"*, i32, i32, %"class.nanojit::Allocator"* }
%68 = type { %struct.JSXMLElemVar }
%69 = type { [36 x i8], [38 x i8], [72 x i8] }
%70 = type { [41 x i8] }
%71 = type { [38 x i8] }
%72 = type { [7 x i8], [15 x i8] }
%73 = type { [8 x i8], [12 x i8], [16 x i8], [21 x i8], [26 x i8], [31 x i8], [36 x i8], [41 x i8], [46 x i8] }
%74 = type { [7 x i8], [7 x i8], [7 x i8], [7 x i8], [7 x i8], [7 x i8], [7 x i8], [7 x i8], [7 x i8] }
%75 = type { [4 x i8] }
%76 = type { [5 x i8] }
%77 = type { i32, void ()* }
%78 = type { i8*, i32, i32, %"class.nanojit::Allocator"* }
%79 = type { i32*, i32, i32, %"class.nanojit::Allocator"* }
%80 = type { i32, i1 }
%class.DSTOffsetCache = type { i64, i64, i64, i64, i64, i64 }
%"class.JSC::ExecutableAllocator" = type { %0 }
%"class.JSC::ExecutablePool" = type { i32, i8, i32, i8*, i8*, %1 }
%"class.JSC::MacroAssemblerCodePtr" = type { i8* }
%"class.JSC::MacroAssemblerCodeRef" = type { %"class.JSC::MacroAssemblerCodePtr", %"class.JSC::ExecutablePool"*, i32 }
%"class.avmplus::AvmConsole" = type { i8 }
%"class.avmplus::AvmCore" = type { %"class.avmplus::AvmInterpreter", %"class.avmplus::AvmConsole" }
%"class.avmplus::AvmInterpreter" = type { %"class.avmplus::AvmConsole", %"class.avmplus::AvmConsole"* }
%"class.avmplus::BitSet" = type { i32, i32* }
%"class.js::AutoGCRooter" = type { %"class.js::AutoGCRooter"*, i32, %struct.JSContext* }
%"class.js::AutoIdRooter" = type { [12 x i8], i32 }
%"class.js::AutoIdVector" = type { [60 x i8] }
%"class.js::AutoValueRooter" = type { [12 x i8], [4 x i8], %"class.js::Value" }
%"class.js::Bindings" = type { %"struct.js::Shape"*, i16, i16, i16 }
%"class.js::BoxArg" = type { %"class.js::TraceRecorder"*, %"struct.js::tjit::Address" }
%"class.js::CaptureTypesVisitor" = type { %struct.JSContext*, i8*, i8*, %"class.js::Oracle"* }
%"class.js::ClearSlotsVisitor" = type { %"class.js::Tracker"* }
%"class.js::ContextAllocPolicy" = type { %struct.JSContext* }
%"class.js::DefaultSlotMap" = type { [28 x i8] }
%"class.js::DetermineTypesVisitor" = type { %"class.js::TraceRecorder"*, i8* }
%"class.js::DtoaCache" = type { double, i32, %struct.JSString* }
%"class.js::FlushNativeStackFrameVisitor" = type { %struct.JSContext*, i8*, i8*, double* }
%"class.js::FrameInfoCache" = type { %29, %"class.js::VMAllocator"* }
%"class.js::GCChunkAllocator" = type { i32 (...)** }
%"class.js::GCHelperThread" = type { %struct.PRThread*, %struct.PRCondVar*, %struct.PRCondVar*, i8, i8, %60, i8**, i8** }
%"class.js::HashMap" = type { %45 }
%"class.js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry" = type { %"class.nanojit::LIns"*, %struct.JSObject* }
%"class.js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry" = type { i8*, %"class.js::LoopProfile"* }
%"class.js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry" = type { i8*, i32 }
%"class.js::HashMap<void *, JSThread *, js::DefaultHasher<void *>, js::SystemAllocPolicy>::Entry" = type { i8*, %struct.JSThread* }
%"class.js::HashSet" = type { %"class.js::detail::HashTable" }
%"class.js::ImportBoxedStackSlotVisitor" = type { %"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, i32, i8*, %struct.JSStackFrame* }
%"class.js::KidsPointer" = type { i32 }
%"class.js::LoopProfile" = type { %"struct.js::TraceMonitor"*, %struct.JSScript*, %struct.JSStackFrame*, i8*, i8*, i32, i8, i8, i8, i8, i8, [11 x i32], i32, [11 x i32], i32, double, double, i8, i8, [8 x %"struct.js::LoopProfile::InnerLoop"], i32, [8 x %"struct.js::LoopProfile::InnerLoop"], i32, [6 x %"struct.js::LoopProfile::StackValue"], i32 }
%"class.js::MathCache" = type { [4096 x %"struct.js::MathCache::Entry"] }
%"class.js::NativeIterCache" = type { [256 x %struct.JSObject*], %struct.JSObject* }
%"class.js::Oracle" = type { %"class.avmplus::BitSet", %"class.avmplus::BitSet", %"class.avmplus::BitSet", %"class.avmplus::BitSet" }
%"class.js::PropertyCache" = type { [4096 x %"struct.js::PropertyCacheEntry"], i32 }
%"class.js::PropertyTree" = type { %struct.JSCompartment*, %struct.JSArenaPool, %"struct.js::Shape"* }
%"class.js::Queue" = type { %"struct.js::TreeFragment"**, i32, i32, %"class.nanojit::Allocator"* }
%"class.js::SlotMap" = type { i32 (...)**, %"class.js::TraceRecorder"*, %struct.JSContext*, %67 }
%"class.js::StackSegment" = type { %struct.JSContext*, %"class.js::StackSegment"*, %"class.js::StackSegment"*, %struct.JSStackFrame*, %struct.JSFrameRegs*, %struct.JSObject*, i8, i8* }
%"class.js::StackSpace" = type { %"class.js::Value"*, %"class.js::Value"*, %"class.js::StackSegment"*, %"class.js::Value"* }
%"class.js::TraceRecorder" = type { %struct.JSContext*, %"struct.js::TraceMonitor"*, %"class.js::Oracle"*, %"class.js::VMFragment"*, %"struct.js::TreeFragment"*, %struct.JSObject*, %struct.JSScript*, i8*, i32, %"struct.js::VMSideExit"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, i32, i32, %"class.js::TypeMap", %"class.nanojit::LirBuffer"*, %"struct.js::VMAllocator::Mark", i32, %"class.js::Tracker", %"class.js::Tracker", %"class.js::Value"*, i32, %struct.JSAtom**, %"class.js::Value"*, %"class.nanojit::LIns"*, %31, i8, %"class.js::Queue", %32, i32, i8, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %struct.JSSpecializedNative*, %"class.js::Value"*, %"class.nanojit::LIns"*, %34, i8, %struct.JSSpecializedNative, %35, %"class.js::tjit::Writer" }
%"class.js::Tracker" = type { %"struct.js::Tracker::TrackerPage"* }
%"class.js::TypeMap" = type { [16 x i8], %"class.js::Oracle"* }
%"class.js::VMAllocator" = type { [12 x i8], i8, i32, i8*, i32, i32 }
%"class.js::VMFragment" = type { [28 x i8], %"struct.js::TreeFragment"* }
%"class.js::Value" = type { %"union.js::AlignedStorage<1>::U" }
%"class.js::Vector" = type { %struct.JSGenerator**, i32, i32, %"class.js::Value" }
%"class.js::detail::HashTable" = type { i32, i32, i32, i32, i32, %"class.js::detail::HashTable<JSObject *const, js::HashSet<JSObject *, js::DefaultHasher<JSObject *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%"class.js::detail::HashTable<JSObject *const, js::HashSet<JSObject *, js::DefaultHasher<JSObject *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::AddPtr" = type { [4 x i8], i32 }
%"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry" = type { i32, %struct.JSScript* }
%"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Ptr" = type { %"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%"class.js::detail::HashTable<const unsigned long, js::HashSet<unsigned long, js::AtomHasher, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::detail::HashTable<js::EmptyShape *const, js::HashSet<js::EmptyShape *, js::DefaultHasher<js::EmptyShape *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::detail::HashTable<js::FrameInfo *const, js::HashSet<js::FrameInfo *, js::FrameInfoCache::HashPolicy, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry" = type { i32, %"struct.js::FrameInfo"* }
%"class.js::detail::HashTable<js::FrameInfo *const, js::HashSet<js::FrameInfo *, js::FrameInfoCache::HashPolicy, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Ptr" = type { %"class.js::detail::HashTable<js::FrameInfo *const, js::HashSet<js::FrameInfo *, js::FrameInfoCache::HashPolicy, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* }
%"class.js::detail::HashTable<js::HashMap<JSFunction *, JSString *, js::DefaultHasher<JSFunction *>, js::SystemAllocPolicy>::Entry, js::HashMap<JSFunction *, JSString *, js::DefaultHasher<JSFunction *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::detail::HashTable<js::HashMap<js::Value, js::Value, js::WrapperHasher, js::SystemAllocPolicy>::Entry, js::HashMap<js::Value, js::Value, js::WrapperHasher, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::detail::HashTable<js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry, js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::MapHashPolicy, js::ContextAllocPolicy>::Entry" = type { i32, %"class.js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry" }
%"class.js::detail::HashTable<js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry, js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::MapHashPolicy, js::ContextAllocPolicy>::Ptr" = type { %"class.js::detail::HashTable<js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry, js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::MapHashPolicy, js::ContextAllocPolicy>::Entry"* }
%"class.js::detail::HashTable<js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry" = type { i32, %"class.js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry" }
%"class.js::detail::HashTable<js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Ptr" = type { %"class.js::detail::HashTable<js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%"class.js::detail::HashTable<js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry" = type { i32, %"class.js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry" }
%"class.js::detail::HashTable<js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Ptr" = type { %"class.js::detail::HashTable<js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry"* }
%"class.js::detail::HashTable<js::HashMap<void *, JSThread *, js::DefaultHasher<void *>, js::SystemAllocPolicy>::Entry, js::HashMap<void *, JSThread *, js::DefaultHasher<void *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry" = type { i32, %"class.js::HashMap<void *, JSThread *, js::DefaultHasher<void *>, js::SystemAllocPolicy>::Entry" }
%"class.js::detail::HashTable<js::HashMap<void *, js::RootInfo, js::DefaultHasher<void *>, js::SystemAllocPolicy>::Entry, js::HashMap<void *, js::RootInfo, js::DefaultHasher<void *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::detail::HashTable<js::HashMap<void *, unsigned int, js::GCPtrHasher, js::SystemAllocPolicy>::Entry, js::HashMap<void *, unsigned int, js::GCPtrHasher, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::detail::HashTable<js::gc::Chunk *const, js::HashSet<js::gc::Chunk *, js::GCChunkHasher, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry" = type opaque
%"class.js::mjit::JaegerCompartment" = type { %"class.JSC::ExecutableAllocator"*, %"struct.js::mjit::Trampolines", %"struct.js::VMFrame"* }
%"class.js::tjit::Writer" = type { %"class.nanojit::Allocator"*, %"class.nanojit::LirBuffer"*, %"class.nanojit::LirFilter"*, %"class.nanojit::CseFilter"*, %"class.nanojit::LogControl"* }
%"class.nanojit::AR" = type { i32, [4096 x %"class.nanojit::LIns"*] }
%"class.nanojit::Allocator" = type { %"class.nanojit::Allocator::Chunk"*, i8*, i8* }
%"class.nanojit::Allocator::Chunk" = type { %"class.nanojit::Allocator::Chunk"*, [1 x i64] }
%"class.nanojit::Assembler" = type { %"class.nanojit::Allocator"*, %"class.nanojit::CodeAlloc"*, %"class.nanojit::Allocator"*, %"class.nanojit::Fragment"*, %21, %23, %"class.nanojit::LabelStateMap", %"class.js::GCChunkAllocator"*, %27, %"class.nanojit::CodeList"*, i8, [3 x i8], i8*, i8*, i8*, i8*, i8*, i8*, i8*, i32, %"class.nanojit::LIns"*, %"class.nanojit::AR", %"class.nanojit::RegAlloc", i32, %"struct.nanojit::Config"* }
%"class.nanojit::BitSet" = type { %"class.nanojit::Allocator"*, i32, i64* }
%"class.nanojit::CodeAlloc" = type { %"class.nanojit::CodeList"*, %"class.nanojit::CodeList"*, i32, i32, i32 }
%"class.nanojit::CodeList" = type { %"class.nanojit::CodeList"*, %"class.nanojit::CodeList"*, %"class.nanojit::CodeList"*, i8, i8, %20, [1 x i8] }
%"class.nanojit::CseFilter" = type { [8 x i8], [8 x %"class.nanojit::LIns"**], [8 x i32], [8 x i32], [8 x %struct.JSObjectMap], i8, i8, i8, i8, [34 x %"class.nanojit::LIns"**], [34 x i32], [34 x i32], i32, %"class.nanojit::Allocator"*, %"class.nanojit::HashMap", i8, i8 }
%"class.nanojit::Fragment" = type { %"class.nanojit::LirBuffer"*, %"class.nanojit::LIns"*, i8*, i32, i8*, i8*, i32 }
%"class.nanojit::HashMap" = type { %"class.nanojit::Allocator"*, i32, %"class.nanojit::Seq"** }
%"class.nanojit::LIns" = type { %"class.JSC::MacroAssemblerCodePtr" }
%"class.nanojit::LInsC" = type { %"class.nanojit::LIns"**, %"struct.nanojit::CallInfo"*, %"class.nanojit::LIns" }
%"class.nanojit::LInsI" = type { i32, %"class.nanojit::LIns" }
%"class.nanojit::LInsOp2" = type { %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns" }
%"class.nanojit::LInsOp3" = type { %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns" }
%"class.nanojit::LInsQorD" = type { i32, i32, %"class.nanojit::LIns" }
%"class.nanojit::LInsSt" = type { i16, i8, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns" }
%"class.nanojit::LabelStateMap" = type { %"class.nanojit::Allocator"*, %25 }
%"class.nanojit::LirBuffer" = type { %"class.js::KidsPointer", i32, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, [4 x %"class.nanojit::LIns"*], %"class.nanojit::Allocator"*, i32, i32 }
%"class.nanojit::LirFilter" = type { i32 (...)**, %"class.nanojit::LirFilter"* }
%"class.nanojit::LogControl" = type { i32 (...)**, i32 }
%"class.nanojit::RegAlloc" = type { [17 x %"class.nanojit::LIns"*], [17 x i32], i32, i32, i32 }
%"class.nanojit::Seq" = type opaque
%"class.nanojit::StackFilter" = type { [8 x i8], %"class.nanojit::LIns"*, %"class.nanojit::BitSet", i32 }
%struct.DtoaState = type opaque
%struct.JSArena = type { %struct.JSArena*, i32, i32, i32 }
%struct.JSArenaPool = type { %struct.JSArena, %struct.JSArena*, i32, i32, i32* }
%struct.JSArgumentFormatMap = type { i8*, i32, i32 (%struct.JSContext*, i8*, i32, i64**, i8**)*, %struct.JSArgumentFormatMap* }
%struct.JSAtom = type { [16 x i8] }
%struct.JSAtomMap = type { %struct.JSAtom**, i32 }
%struct.JSAtomState = type { %64, %struct.JSThinLock, %struct.JSAtom*, [2 x %struct.JSAtom*], [8 x %struct.JSAtom*], %struct.JSAtom*, [39 x %struct.JSAtom*], %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %struct.JSAtom*, %66 }
%struct.JSCListStr = type { %struct.JSCListStr*, %struct.JSCListStr* }
%struct.JSCodeSpec = type { i8, i8, i8, i8, i32 }
%struct.JSCompartment = type { %struct.JSRuntime*, %struct.JSPrincipals*, %"struct.js::gc::Chunk"*, [11 x %"struct.js::gc::ArenaList"], %"struct.js::gc::FreeLists", i32, i32, i32, %"struct.js::TraceMonitor", [64 x %struct.JSScript*], i8*, i8, %"class.js::HashMap", %"class.js::mjit::JaegerCompartment"*, %"class.js::PropertyTree", %"struct.js::EmptyShape"*, %"struct.js::EmptyShape"*, %"struct.js::EmptyShape"*, %"struct.js::EmptyShape"*, %"struct.js::EmptyShape"*, %"struct.js::EmptyShape"*, %47, i8, %struct.JSCListStr, %"class.JSC::ExecutableAllocator"*, %"class.js::NativeIterCache", %49, %"class.js::DtoaCache", %"class.js::MathCache"*, i8, %39 }
%struct.JSContext = type { %struct.JSCListStr, i32, i32, i8, i32, %"class.js::Value", i32, %struct.JSLocaleCallbacks*, %struct.JSDHashTable*, i8, i32, i32, %struct.JSRuntime*, %struct.JSCompartment*, %struct.JSFrameRegs*, %struct.JSArenaPool, %struct.JSArenaPool, %struct.JSObject*, %struct.JSSharpObjectMap, %"class.js::HashSet", %struct.JSArgumentFormatMap*, i8*, void (%struct.JSContext*, i8*, %struct.JSErrorReport*)*, i32 (%struct.JSContext*)*, i32, i8*, i8*, %"class.js::StackSegment"*, %struct.JSThread*, i32, %struct.JSCListStr, %"class.js::AutoGCRooter"*, %struct.JSDebugHooks*, %struct.JSSecurityCallbacks*, i32, i64, [4 x i8], %"class.js::Value", i8, i8, i8, %class.DSTOffsetCache, %struct.JSObject*, %"class.js::Vector", %"class.js::GCHelperThread"* }
%struct.JSDHashTable = type { %struct.JSDHashTableOps*, i8*, i16, i8, i8, i32, i32, i32, i32, i8* }
%struct.JSDHashTableOps = type { i8* (%struct.JSDHashTable*, i32)*, void (%struct.JSDHashTable*, i8*)*, i32 (%struct.JSDHashTable*, i8*)*, i32 (%struct.JSDHashTable*, %"class.js::KidsPointer"*, i8*)*, void (%struct.JSDHashTable*, %"class.js::KidsPointer"*, %"class.js::KidsPointer"*)*, void (%struct.JSDHashTable*, %"class.js::KidsPointer"*)*, void (%struct.JSDHashTable*)*, i32 (%struct.JSDHashTable*, %"class.js::KidsPointer"*, i8*)* }
%struct.JSDebugHooks = type { i32 (%struct.JSContext*, %struct.JSScript*, i8*, i64*, i8*)*, i8*, void (%struct.JSContext*, i8*, i32, %struct.JSScript*, %struct.JSFunction*, i8*)*, i8*, void (%struct.JSContext*, %struct.JSScript*, i8*)*, i8*, i32 (%struct.JSContext*, %struct.JSScript*, i8*, i64*, i8*)*, i8*, void (i8*, i32, i16*, i32, i8**, i8*)*, i8*, i8* (%struct.JSContext*, %struct.JSStackFrame*, i32, i32*, i8*)*, i8*, i8* (%struct.JSContext*, %struct.JSStackFrame*, i32, i32*, i8*)*, i8*, i32 (%struct.JSContext*, %struct.JSScript*, i8*, i64*, i8*)*, i8*, i32 (%struct.JSContext*, i8*, %struct.JSErrorReport*, i8*)*, i8* }
%struct.JSErrorFormatString = type { i8*, i16, i16 }
%struct.JSErrorReport = type { i8*, i32, i8*, i8*, i16*, i16*, i32, i32, i16*, i16** }
%struct.JSFatLock = type opaque
%struct.JSFrameRegs = type { %"class.js::Value"*, i8*, %struct.JSStackFrame* }
%struct.JSFunction = type { [56 x i8], i16, i16, %"union.JSFunction::U", %struct.JSAtom*, [4 x i8] }
%struct.JSGSNCache = type { i8*, %struct.JSDHashTable }
%struct.JSGenerator = type { %struct.JSObject*, i32, %struct.JSFrameRegs, %struct.JSObject*, %struct.JSStackFrame*, [4 x i8], [1 x %"class.js::Value"] }
%struct.JSHashAllocOps = type { i8* (i8*, i32)*, void (i8*, i8*, i32)*, %struct.JSHashEntry* (i8*, i8*)*, void (i8*, %struct.JSHashEntry*, i32)* }
%struct.JSHashEntry = type { %struct.JSHashEntry*, i32, i8*, i8* }
%struct.JSHashTable = type { %struct.JSHashEntry**, i32, i32, i32 (i8*)*, i32 (i8*, i8*)*, i32 (i8*, i8*)*, %struct.JSHashAllocOps*, i8* }
%struct.JSLocaleCallbacks = type { i32 (%struct.JSContext*, %struct.JSString*, i64*)*, i32 (%struct.JSContext*, %struct.JSString*, i64*)*, i32 (%struct.JSContext*, %struct.JSString*, %struct.JSString*, i64*)*, i32 (%struct.JSContext*, i8*, i64*)*, %struct.JSErrorFormatString* (i8*, i8*, i32)* }
%struct.JSNativeTraceInfo = type { i32 (%struct.JSContext*, i32, %"class.js::Value"*)*, %struct.JSSpecializedNative* }
%struct.JSObject = type { %11, %"struct.js::Class"*, i32, i32, %"struct.js::EmptyShape"**, %struct.JSObject*, %struct.JSObject*, i8*, i32, %"class.js::Value"* }
%struct.JSObjectMap = type { i32, i32 }
%struct.JSPendingProxyOperation = type { %struct.JSPendingProxyOperation*, %struct.JSObject* }
%struct.JSPrincipals = type { i8*, i8* (%struct.JSContext*, %struct.JSPrincipals*)*, i32 (%struct.JSContext*, %struct.JSPrincipals*)*, i32, void (%struct.JSContext*, %struct.JSPrincipals*)*, i32 (%struct.JSPrincipals*, %struct.JSPrincipals*)* }
%struct.JSProperty = type opaque
%struct.JSRuntime = type { %struct.JSCompartment*, i8, %51, i32, i32 (%struct.JSContext*, i32)*, i32 (%struct.JSContext*, %struct.JSCompartment*, i32)*, void (i8*, i32)*, i8*, i32, %52, %54, %56, i32, i32, i32, i32, i32, i32, i32, i32, i32, %"struct.js::GCMarker"*, i32, i64, i32, i8, %struct.JSCompartment*, %struct.JSCompartment*, i8, i8, i8, i8, i32 (%struct.JSContext*, i32)*, i32, %"class.js::GCChunkAllocator"*, void (%struct.JSTracer*, i8*)*, i8*, %"class.js::Value", %"class.js::Value", %"class.js::Value", %struct.JSAtom*, %struct.JSCListStr, %struct.JSDebugHooks, i32, %struct.JSCListStr, %struct.JSCListStr, i8*, %struct.PRLock*, %struct.PRCondVar*, %struct.PRCondVar*, i32, %struct.JSThread*, %"class.js::GCHelperThread", %struct.PRLock*, %struct.PRCondVar*, %struct.PRLock*, %62, i32, %struct.JSSecurityCallbacks*, %struct.JSStructuredCloneCallbacks*, i32, %struct.JSHashTable*, %struct.JSCListStr, %struct.PRLock*, i8*, i8*, i8*, %struct.JSObject*, %struct.JSObject*, i32, i32, %struct.JSAtomState, %struct.JSObject* (%struct.JSContext*, %struct.JSObject*, %struct.JSObject*, %struct.JSObject*, i32)*, %struct.JSObject* (%struct.JSContext*, %struct.JSObject*, %struct.JSObject*, i32)*, i32, i32, [4 x i8] }
%struct.JSScript = type { %struct.JSCListStr, i8*, i32, i16, i32, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8*, %struct.JSAtomMap, %struct.JSCompartment*, i8*, i32, i16, i16, i16, i16, %"class.js::Bindings", %struct.JSPrincipals*, %15, i32*, i8*, i8*, %"struct.js::mjit::JITScript"*, %"struct.js::mjit::JITScript"* }
%struct.JSSecurityCallbacks = type { i32 (%struct.JSContext*, %struct.JSObject*, i32, i32, i64*)*, i32 (%struct.JSXDRState*, %struct.JSPrincipals**)*, %struct.JSPrincipals* (%struct.JSContext*, %struct.JSObject*)*, i32 (%struct.JSContext*)* }
%struct.JSSharpObjectMap = type { i32, i32, %struct.JSHashTable* }
%struct.JSSpecializedNative = type { %"struct.nanojit::CallInfo"*, i8*, i8*, i32 }
%struct.JSStackFrame = type { i32, %37, %"class.js::KidsPointer", %struct.JSObject*, %struct.JSStackFrame*, i8*, %"class.js::Value", i8*, i8*, i8*, i8* }
%struct.JSString = type { i32, %3, %4 }
%struct.JSStructuredCloneCallbacks = type { %struct.JSObject* (%struct.JSContext*, %struct.JSStructuredCloneReader*, i32, i32, i8*)*, i32 (%struct.JSContext*, %struct.JSStructuredCloneWriter*, %struct.JSObject*, i8*)*, void (%struct.JSContext*, i32)* }
%struct.JSStructuredCloneReader = type opaque
%struct.JSStructuredCloneWriter = type opaque
%struct.JSThinLock = type { i32, %struct.JSFatLock* }
%struct.JSThread = type { %struct.JSCListStr, i8*, i32, %struct.JSThreadData }
%struct.JSThreadData = type { i32, %struct.JSCompartment*, %struct.JSCompartment*, %struct.JSCompartment*, i32, %"class.js::StackSpace", i8, %struct.JSGSNCache, %"class.js::PropertyCache", i32, %struct.DtoaState*, i32*, %struct.JSPendingProxyOperation*, %"struct.js::ConservativeGCThreadData" }
%struct.JSTracer = type { %struct.JSContext*, void (%struct.JSTracer*, i8*, i32)*, void (%struct.JSTracer*, i8*, i32)*, i8*, i32 }
%struct.JSUpvarArray = type { %"class.js::KidsPointer"*, i32 }
%struct.JSXDRState = type opaque
%struct.JSXML = type { %struct.JSObject*, i8*, %struct.JSXML*, %struct.JSObject*, i32, i32, %68 }
%struct.JSXMLArray = type { i32, i32, i8**, %struct.JSXMLArrayCursor* }
%struct.JSXMLArrayCursor = type { %struct.JSXMLArray*, i32, %struct.JSXMLArrayCursor*, %struct.JSXMLArrayCursor**, i8* }
%struct.JSXMLElemVar = type { %struct.JSXMLArray, %struct.JSXMLArray, %struct.JSXMLArray }
%struct.PRCondVar = type opaque
%struct.PRLock = type opaque
%struct.PRThread = type opaque
%struct.anon = type { [36 x i8], [38 x i8] }
%"struct.js::ArgumentsData" = type { %"class.js::Value", [1 x %"class.js::Value"] }
%"struct.js::Class" = type { i8*, i32, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, %struct.JSObject*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)*, void (%struct.JSContext*, %struct.JSObject*)*, void ()*, i32 (%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, i32, %"class.js::Value"*)*, i32 (%struct.JSXDRState*, %struct.JSObject**)*, i32 (%struct.JSContext*, %struct.JSObject*, %"class.js::Value"*, i32*)*, i32 (%struct.JSContext*, %struct.JSObject*, i8*)*, %"struct.js::ClassExtension", %"struct.js::ObjectOps", [8 x i8] }
%"struct.js::ClassExtension" = type { i32 (%struct.JSContext*, %struct.JSObject*, %"class.js::Value"*, i32*)*, %struct.JSObject* (%struct.JSContext*, %struct.JSObject*)*, %struct.JSObject* (%struct.JSContext*, %struct.JSObject*)*, %struct.JSObject* (%struct.JSContext*, %struct.JSObject*, i32)*, i8* }
%"struct.js::ConservativeGCThreadData" = type { i32*, %59, i32 }
%"struct.js::CountSlotsVisitor" = type { i32, i8, i8* }
%"struct.js::EmptyShape" = type { [40 x i8] }
%"struct.js::FrameInfo" = type { %struct.JSObject*, i8*, i8*, i32, i32, i32, i32 }
%"struct.js::GCMarker" = type { [20 x i8], i32, i32, %9* }
%"struct.js::GlobalState" = type { %struct.JSObject*, i32, %38* }
%"struct.js::LinkableFragment" = type { [32 x i8], i32, %"class.js::TypeMap", i32, i32, %38* }
%"struct.js::LoopProfile::InnerLoop" = type { %struct.JSStackFrame*, i8*, i8*, i32 }
%"struct.js::LoopProfile::StackValue" = type { i8, i8, i32 }
%"struct.js::MathCache::Entry" = type { double, double (double)*, double }
%"struct.js::ObjectOps" = type { i32 (%struct.JSContext*, %struct.JSObject*, i32, %struct.JSObject**, %struct.JSProperty**)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)*, i32)*, i32 (%struct.JSContext*, %struct.JSObject*, %struct.JSObject*, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, i32*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, i32*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32*)*, i32 (%struct.JSContext*, %struct.JSObject*)*, void (%struct.JSTracer*, %struct.JSObject*)*, i32 (%struct.JSContext*, %struct.JSObject*, i8*, %"class.js::AutoIdVector"*)*, %struct.JSObject* (%struct.JSContext*, %struct.JSObject*)*, void (%struct.JSContext*, %struct.JSObject*)* }
%"struct.js::PICTable" = type { [32 x %"struct.js::PICTableEntry"], i32 }
%"struct.js::PICTableEntry" = type { i32, i32, i32 }
%"struct.js::PropertyCacheEntry" = type { i8*, i32, i32, %"class.js::KidsPointer" }
%"struct.js::PropertyTable" = type { i32, i32, i32, i32, %"struct.js::Shape"** }
%"struct.js::Shape" = type { [8 x i8], %"class.js::KidsPointer", i32, %12, %13, i32, i8, i8, i16, %"struct.js::Shape"*, %14 }
%"struct.js::SlotMap::SlotInfo" = type { i8*, i8, i32, i8 }
%"struct.js::TraceMonitor" = type { %struct.JSContext*, %"struct.js::TracerState"*, %"struct.js::VMSideExit"*, i32, %"struct.js::TraceNativeStorage"*, %"class.js::VMAllocator"*, %"class.js::VMAllocator"*, %"class.js::VMAllocator"*, %"class.nanojit::CodeAlloc"*, %"class.nanojit::Assembler"*, %"class.js::FrameInfoCache"*, i32, %"class.js::Oracle"*, %"class.js::TraceRecorder"*, %"class.js::LoopProfile"*, [4 x %"struct.js::GlobalState"], [512 x %"struct.js::TreeFragment"*], %39*, %41*, i32, i32, %"class.js::TypeMap"*, %43 }
%"struct.js::TraceNativeStorage" = type { [8193 x double], [500 x %"struct.js::FrameInfo"*] }
%"struct.js::TraceRecorder::NameResult" = type { i8, [7 x i8], %"class.js::Value", %struct.JSObject*, %"class.nanojit::LIns"*, %"struct.js::Shape"*, [4 x i8] }
%"struct.js::TracerState" = type { %struct.JSContext*, %"struct.js::TraceMonitor"*, double*, double*, double*, %"struct.js::FrameInfo"**, i8*, %"struct.js::FrameInfo"**, i8*, %"struct.js::VMSideExit"*, %"struct.js::VMSideExit"*, i8*, %"struct.js::VMSideExit"*, %"struct.js::TreeFragment"*, i32*, %"struct.js::VMSideExit"**, %"struct.js::VMSideExit"*, i64, %"struct.js::TracerState"*, i32, double*, i32, %"class.js::Value"* }
%"struct.js::Tracker::TrackerPage" = type { %"struct.js::Tracker::TrackerPage"*, i32, [1024 x %"class.nanojit::LIns"*] }
%"struct.js::TreeFragment" = type { [68 x i8], %"struct.js::TreeFragment"*, %"struct.js::TreeFragment"*, %"struct.js::TreeFragment"*, %struct.JSObject*, i32, i32, %"class.js::Queue", %"class.js::Queue", %struct.JSScript*, %"struct.js::UnstableExit"*, %17, i32, i32, %18, %19, i32, i32, i32 }
%"struct.js::TypedArray" = type { %"class.js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry"*, %struct.JSObject*, i32, i32, i32, i32, i8* }
%"struct.js::UnstableExit" = type { %"class.js::VMFragment"*, %"struct.js::VMSideExit"*, %"struct.js::UnstableExit"* }
%"struct.js::VMAllocator::Mark" = type { %"class.js::VMAllocator"*, i8, %"class.nanojit::Allocator::Chunk"*, i8*, i8*, i32 }
%"struct.js::VMFrame" = type { %"union.js::VMFrame::Arguments", %"struct.js::VMFrame"*, i8*, %struct.JSFrameRegs, %struct.JSContext*, %"class.js::Value"*, %struct.JSStackFrame*, i8*, i8*, i8*, i8*, i8* }
%"struct.js::VMSideExit" = type { [16 x i8], i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%"struct.js::gc::Arena" = type { %7, %"struct.js::gc::Things" }
%"struct.js::gc::ArenaBitmap" = type { [16 x i32] }
%"struct.js::gc::ArenaHeader" = type { %struct.JSCompartment*, %"struct.js::gc::Arena"*, %"struct.js::gc::FreeCell"*, i32, i8, i32 }
%"struct.js::gc::ArenaList" = type { %"struct.js::gc::Arena"*, %"struct.js::gc::Arena"* }
%"struct.js::gc::Chunk" = type { [251 x %"struct.js::gc::Arena"], [251 x %"struct.js::gc::ArenaBitmap"], [251 x %"struct.js::gc::MarkingDelay"], %"struct.js::gc::ChunkInfo" }
%"struct.js::gc::ChunkInfo" = type { %"struct.js::gc::Chunk"*, %struct.JSRuntime*, %"struct.js::gc::EmptyArenaLists", i32, i32 }
%"struct.js::gc::EmptyArenaLists" = type { %"struct.js::gc::Arena"*, [11 x %"struct.js::gc::Arena"*] }
%"struct.js::gc::FreeCell" = type { %8 }
%"struct.js::gc::FreeLists" = type { [11 x %"struct.js::gc::FreeCell"**] }
%"struct.js::gc::MarkingDelay" = type { %9*, i32, i32 }
%"struct.js::gc::Things" = type { [509 x %"union.js::gc::ThingOrCell"] }
%"struct.js::mjit::JITScript" = type { %"class.JSC::MacroAssemblerCodeRef", i8*, i8*, i8*, [4 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, %16 }
%"struct.js::mjit::Trampolines" = type { void ()*, %"class.JSC::ExecutablePool"* }
%"struct.js::tjit::Address" = type { %"class.nanojit::LIns"*, i32, i32 }
%"struct.nanojit::CallInfo" = type { i32, [4 x i8], i32 }
%"struct.nanojit::Config" = type { i8, i8, i8, i8 }
%"struct.nanojit::GuardRecord" = type { i8*, %"struct.nanojit::GuardRecord"*, %"struct.nanojit::SideExit"* }
%"struct.nanojit::Interval" = type { i64, i64, i8 }
%"struct.nanojit::SideExit" = type { %"struct.nanojit::GuardRecord"*, %"class.nanojit::Fragment"*, %"class.nanojit::Fragment"*, %"struct.nanojit::SwitchInfo"* }
%"struct.nanojit::SwitchInfo" = type { i8**, i32, i32 }
%"union.JSFunction::U" = type { %58 }
%"union.js::AlignedStorage<16>::U" = type { i64, [8 x i8] }
%"union.js::AlignedStorage<1>::U" = type { i64 }
%"union.js::AlignedStorage<256>::U" = type { i64, [248 x i8] }
%"union.js::AlignedStorage<64>::U" = type { i64, [56 x i8] }
%"union.js::VMFrame::Arguments" = type { %46 }
%"union.js::gc::ThingOrCell" = type { %"struct.js::gc::FreeCell" }

@_ZN7nanojitL9SavedRegsE = external global i32, align 4
@_ZN7nanojitL7XmmRegsE = external global i32, align 4
@_ZN7nanojitL7x87RegsE = external global i32, align 4
@_ZN7nanojitL6FpRegsE = external global i32, align 4
@_ZL16equality_imacros = external global %struct.anon, align 1
@_ZL14binary_imacros = external global %69, align 1
@_ZL11add_imacros = external global %69, align 1
@_ZL13unary_imacros = external global %70, align 1
@_ZL12call_imacros = external global %71, align 1
@_ZL11new_imacros = external global %71, align 1
@js_opcode2extra = external global [244 x i8], align 1
@_ZL15incelem_imacros = external global %72, align 1
@_ZL15decelem_imacros = external global %72, align 1
@_ZL16funapply_imacros = external global %73, align 1
@_ZL15funcall_imacros = external global %74, align 1
@_ZL15getprop_imacros = external global %75, align 1
@_ZL16callprop_imacros = external global %76, align 1
@_ZL19getthisprop_imacros = external global %75, align 1
@.str = external constant [46 x i8]
@.str6 = external constant [51 x i8]
@_ZL17js_IntToString_ci = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL6s_coreE = external global %"class.avmplus::AvmCore", align 4
@_ZN2jsL31did_we_check_processor_featuresE.b = external global i1
@_ZN2js13LogControllerE = external global %"class.nanojit::LogControl", align 8
@__dso_handle = external global i8*
@.str10 = external constant [48 x i8]
@.str11 = external constant [44 x i8]
@_ZN7avmplus7AvmCore6configE = external global %"struct.nanojit::Config"
@js_CodeSpec = external global [0 x %struct.JSCodeSpec]
@_ZN2jsL16GetClosureVar_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL16GetClosureArg_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_dmod_ci = external global %"struct.nanojit::CallInfo"
@js_UnboxDouble_ci = external global %"struct.nanojit::CallInfo"
@js_UnboxInt32_ci = external global %"struct.nanojit::CallInfo"
@js_StringToNumber_ci = external global %"struct.nanojit::CallInfo"
@js_StringToInt32_ci = external global %"struct.nanojit::CallInfo"
@js_DoubleToInt32_ci = external global %"struct.nanojit::CallInfo"
@js_DoubleToUint32_ci = external global %"struct.nanojit::CallInfo"
@js_NumberToString_ci = external global %"struct.nanojit::CallInfo"
@js_BooleanIntToString_ci = external global %"struct.nanojit::CallInfo"
@js_EqualStringsOnTrace_ci = external global %"struct.nanojit::CallInfo"
@js_NaN = external global double
@js_CompareStringsOnTrace_ci = external global %"struct.nanojit::CallInfo"
@_ZN7nanojit8retTypesE = external constant [0 x i32]
@js_FunctionClass = external global %"struct.js::Class"
@js_ArrayClass = external global %"struct.js::Class"
@js_PutArgumentsOnTrace_ci = external global %"struct.nanojit::CallInfo"
@js_PutCallObjectOnTrace_ci = external global %"struct.nanojit::CallInfo"
@js_CreateCallObjectOnTrace_ci = external global %"struct.nanojit::CallInfo"
@_ZN2jsL16functionProbe_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_NewArgumentsOnTrace_ci = external global %"struct.nanojit::CallInfo"
@js_ConcatStrings_ci = external global %"struct.nanojit::CallInfo"
@js_String_tn_ci = external global %"struct.nanojit::CallInfo"
@_ZN2js21NewDenseEmptyArray_ciE = external global %"struct.nanojit::CallInfo"
@_ZN2js27NewDenseUnallocatedArray_ciE = external global %"struct.nanojit::CallInfo"
@_ZN2js25NewDenseAllocatedArray_ciE = external global %"struct.nanojit::CallInfo"
@_ZN2jsL19ceilReturningInt_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL20floorReturningInt_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL20roundReturningInt_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_ObjectClass = external global %"struct.js::Class"
@_ZN2jsL15DeleteIntKey_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL15DeleteStrKey_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_TypeOfObject_ci = external global %"struct.nanojit::CallInfo"
@_ZN2jsL21MethodWriteBarrier_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_AddAtomProperty_ci = external global %"struct.nanojit::CallInfo"
@js_AddProperty_ci = external global %"struct.nanojit::CallInfo"
@js_SetCallArg_ci = external global %"struct.nanojit::CallInfo"
@js_SetCallVar_ci = external global %"struct.nanojit::CallInfo"
@_ZN2jsL20GetPropertyByName_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL21GetPropertyByIndex_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL18GetPropertyById_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL30GetPropertyWithNativeGetter_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_Flatten_ci = external global %"struct.nanojit::CallInfo"
@_ZN8JSString15unitStringTableE = external global [0 x %struct.JSString]
@_ZN2jsL20SetPropertyByName_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL21InitPropertyByName_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL21SetPropertyByIndex_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL22InitPropertyByIndex_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_TypedArray_uint8_clamp_double_ci = external global %"struct.nanojit::CallInfo"
@js_EnsureDenseArrayCapacity_ci = external global %"struct.nanojit::CallInfo"
@_ZN2jsL21GetUpvarArgOnTrace_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL21GetUpvarVarOnTrace_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL23GetUpvarStackOnTrace_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_CreateThisFromTrace_ci = external global %"struct.nanojit::CallInfo"
@_ZN2jsL21funapply_imacro_tableE = external constant [9 x i8*], align 4
@_ZN2jsL20funcall_imacro_tableE = external constant [9 x i8*], align 4
@_ZN2jsL20MethodReadBarrier_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_InitializerObject_ci = external global %"struct.nanojit::CallInfo"
@_ZN2jsL19ObjectToIterator_ciE = external global %"struct.nanojit::CallInfo", align 4
@_ZN2jsL15IteratorMore_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_IteratorClass = external global %"struct.js::Class"
@_ZN2jsL16CloseIterator_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_HasNamedPropertyInt32_ci = external global %"struct.nanojit::CallInfo"
@js_HasNamedProperty_ci = external global %"struct.nanojit::CallInfo"
@_ZN2jsL21HasInstanceOnTrace_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_NewNullClosure_ci = external global %"struct.nanojit::CallInfo"
@js_CloneFunctionObject_ci = external global %"struct.nanojit::CallInfo"
@js_AllocFlatClosure_ci = external global %"struct.nanojit::CallInfo"
@js_CloneRegExpObject_ci = external global %"struct.nanojit::CallInfo"
@js_ArrayCompPush_tn_ci = external global %"struct.nanojit::CallInfo"
@js_SlowArrayClass = external global %"struct.js::Class"
@_ZN2jsL13js_Unbrand_ciE = external global %"struct.nanojit::CallInfo", align 4
@js_DateClass = external global %"struct.js::Class"
@js_BooleanClass = external global %"struct.js::Class"
@js_StringClass = external global %"struct.js::Class"
@js_NumberClass = external global %"struct.js::Class"
@js_CallClass = external global %"struct.js::Class"
@js_BlockClass = external global %"struct.js::Class"
@js_DeclEnvClass = external global %"struct.js::Class"
@js_ArgumentsClass = external global %"struct.js::Class"
@js_RegExpClass = external global %"struct.js::Class"
@js_XMLClass = external global %"struct.js::Class"
@_ZN2js20StrictArgumentsClassE = external global %"struct.js::Class"
@_ZN11JSObjectMap15sharedNonNativeE = external global %struct.JSObjectMap
@.str456 = external constant [15 x i8]
@.str457 = external constant [5 x i8]
@.str458 = external constant [2 x i8]
@.str459 = external constant [4 x i8]
@_ZTVN2js14DefaultSlotMapE = external constant [6 x i8*]
@_ZTVN2js7SlotMapE = external constant [6 x i8*]
@_ZN7nanojit8repKindsE = external constant [0 x i8]
@_ZZN2js2gc23GetFinalizableTraceKindEmE3map = external constant [11 x i8], align 1
@_ZN8JSString18hundredStringTableE = external global [0 x %struct.JSString]
@_ZN8JSString18length2StringTableE = external global [0 x %struct.JSString]
@_ZTVN7nanojit10LogControlE = external constant [4 x i8*]
@llvm.global_ctors = external global [1 x %77]
@.memset_pattern = external constant [2 x i64], align 16

declare hidden i8* @_Z17js_GetImacroStartPh(i8*) nounwind readnone

declare hidden i8* @_ZN7nanojit9Allocator10allocChunkEmb(%"class.nanojit::Allocator"* nocapture, i32, i1 zeroext) nounwind align 2

declare void @JS_Assert(i8*, i8*, i32)

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

declare hidden void @_ZN7nanojit9Allocator9freeChunkEPv(%"class.nanojit::Allocator"* nocapture, i8*) nounwind align 2

declare hidden void @_ZN7nanojit9Allocator9postResetEv(%"class.nanojit::Allocator"* nocapture) nounwind align 2

declare hidden i32 @_ZN7nanojit11StackFilter6getTopEPNS_4LInsE(%"class.nanojit::StackFilter"* nocapture, %"class.nanojit::LIns"*) nounwind readonly align 2

declare x86_fastcallcc %struct.JSString* @_Z14js_IntToStringP9JSContexti(%struct.JSContext*, i32)

declare hidden void @_ZN7nanojit10LogControlD1Ev(%"class.nanojit::LogControl"* nocapture) nounwind align 2

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

declare hidden void @_ZN2js7TrackerC1Ev(%"class.js::Tracker"* nocapture) nounwind align 2

declare hidden void @_ZN2js7TrackerC2Ev(%"class.js::Tracker"* nocapture) nounwind align 2

declare hidden void @_ZN2js7TrackerD1Ev(%"class.js::Tracker"* nocapture) nounwind align 2

declare hidden void @_ZN2js7TrackerD2Ev(%"class.js::Tracker"* nocapture) nounwind align 2

declare hidden void @_ZN2js7Tracker5clearEv(%"class.js::Tracker"* nocapture) nounwind align 2

declare hidden %"struct.js::Tracker::TrackerPage"* @_ZNK2js7Tracker15findTrackerPageEPKv(%"class.js::Tracker"* nocapture, i8*) nounwind readonly align 2

declare hidden %"struct.js::Tracker::TrackerPage"* @_ZN2js7Tracker14addTrackerPageEPKv(%"class.js::Tracker"* nocapture, i8*) nounwind align 2

declare hidden zeroext i1 @_ZNK2js7Tracker3hasEPKv(%"class.js::Tracker"* nocapture, i8*) nounwind readonly align 2

declare hidden %"class.nanojit::LIns"* @_ZNK2js7Tracker3getEPKv(%"class.js::Tracker"* nocapture, i8*) nounwind readonly align 2

declare hidden void @_ZN2js7Tracker3setEPKvPN7nanojit4LInsE(%"class.js::Tracker"* nocapture, i8*, %"class.nanojit::LIns"*) nounwind align 2

declare hidden void @_ZN2js6OracleC1Ev(%"class.js::Oracle"* nocapture) nounwind align 2

declare hidden void @_ZN2js6OracleC2Ev(%"class.js::Oracle"* nocapture) nounwind align 2

declare hidden void @_ZN2js6Oracle25markGlobalSlotUndemotableEP9JSContextj(%"class.js::Oracle"* nocapture, %struct.JSContext* nocapture, i32) nounwind align 2

declare hidden zeroext i1 @_ZNK2js6Oracle23isGlobalSlotUndemotableEP9JSContextj(%"class.js::Oracle"* nocapture, %struct.JSContext* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js6Oracle24markStackSlotUndemotableEP9JSContextjPKv(%"class.js::Oracle"* nocapture, %struct.JSContext* nocapture, i32, i8*) nounwind align 2

declare hidden void @_ZN2js6Oracle24markStackSlotUndemotableEP9JSContextj(%"class.js::Oracle"* nocapture, %struct.JSContext* nocapture, i32) nounwind align 2

declare hidden zeroext i1 @_ZNK2js6Oracle22isStackSlotUndemotableEP9JSContextjPKv(%"class.js::Oracle"* nocapture, %struct.JSContext* nocapture, i32, i8*) nounwind readonly align 2

declare hidden zeroext i1 @_ZNK2js6Oracle22isStackSlotUndemotableEP9JSContextj(%"class.js::Oracle"* nocapture, %struct.JSContext* nocapture, i32) nounwind readonly align 2

declare hidden void @_ZN2js6Oracle26markInstructionUndemotableEPh(%"class.js::Oracle"* nocapture, i8*) nounwind align 2

declare hidden zeroext i1 @_ZNK2js6Oracle24isInstructionUndemotableEPh(%"class.js::Oracle"* nocapture, i8*) nounwind readonly align 2

declare hidden void @_ZN2js6Oracle27markInstructionSlowZeroTestEPh(%"class.js::Oracle"* nocapture, i8*) nounwind align 2

declare hidden zeroext i1 @_ZNK2js6Oracle25isInstructionSlowZeroTestEPh(%"class.js::Oracle"* nocapture, i8*) nounwind readonly align 2

declare hidden void @_ZN2js6Oracle17clearDemotabilityEv(%"class.js::Oracle"* nocapture) nounwind align 2

declare hidden void @_ZN2js14FrameInfoCacheC1EPNS_11VMAllocatorE(%"class.js::FrameInfoCache"* nocapture, %"class.js::VMAllocator"*) nounwind align 2

declare hidden void @_ZN2js14FrameInfoCacheC2EPNS_11VMAllocatorE(%"class.js::FrameInfoCache"* nocapture, %"class.js::VMAllocator"*) nounwind align 2

declare hidden void @_ZN2js12TreeFragment10initializeEP9JSContextPNS_5QueueItEEb(%"struct.js::TreeFragment"* nocapture, %struct.JSContext*, %38*, i1 zeroext) nounwind align 2

declare hidden void @_ZN2js7TypeMap12captureTypesEP9JSContextP8JSObjectRNS_5QueueItEEjb(%"class.js::TypeMap"* nocapture, %struct.JSContext*, %struct.JSObject* nocapture, %38* nocapture, i32, i1 zeroext) nounwind align 2

declare hidden %"struct.js::UnstableExit"* @_ZN2js12TreeFragment18removeUnstableExitEPNS_10VMSideExitE(%"struct.js::TreeFragment"* nocapture, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden i32 @_ZN2js16NativeStackSlotsEP9JSContextj(%struct.JSContext* nocapture, i32) nounwind readonly

declare hidden void @_ZN2js7TypeMap3setEjjPK11JSValueTypeS3_(%"class.js::TypeMap"* nocapture, i32, i32, i8* nocapture, i8* nocapture) nounwind align 2

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

declare hidden void @_ZN2js7TypeMap25captureMissingGlobalTypesEP9JSContextP8JSObjectRNS_5QueueItEEjb(%"class.js::TypeMap"* nocapture, %struct.JSContext*, %struct.JSObject* nocapture, %38* nocapture, i32, i1 zeroext) nounwind align 2

declare hidden zeroext i1 @_ZNK2js7TypeMap7matchesERS0_(%"class.js::TypeMap"* nocapture, %"class.js::TypeMap"* nocapture) nounwind readonly align 2

declare i32 @memcmp(i8* nocapture, i8* nocapture, i32) nounwind readonly

declare hidden void @_ZN2js7TypeMap7fromRawEP11JSValueTypej(%"class.js::TypeMap"* nocapture, i8* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js13FlushJITCacheEP9JSContextPNS_12TraceMonitorE(%struct.JSContext* nocapture, %"struct.js::TraceMonitor"* nocapture) nounwind

declare hidden void @_ZN2js13TraceRecorderC1EP9JSContextPNS_12TraceMonitorEPNS_10VMSideExitEPNS_10VMFragmentEjjP11JSValueTypeS6_P8JSScriptPhjb(%"class.js::TraceRecorder"*, %struct.JSContext*, %"struct.js::TraceMonitor"*, %"struct.js::VMSideExit"*, %"class.js::VMFragment"*, i32, i32, i8*, %"struct.js::VMSideExit"*, %struct.JSScript*, i8*, i32, i1 zeroext) nounwind align 2

declare hidden void @_ZN2js13TraceRecorderC2EP9JSContextPNS_12TraceMonitorEPNS_10VMSideExitEPNS_10VMFragmentEjjP11JSValueTypeS6_P8JSScriptPhjb(%"class.js::TraceRecorder"*, %struct.JSContext*, %"struct.js::TraceMonitor"*, %"struct.js::VMSideExit"*, %"class.js::VMFragment"*, i32, i32, i8*, %"struct.js::VMSideExit"*, %struct.JSScript*, i8*, i32, i1 zeroext) nounwind align 2

declare void @_ZN7nanojit9LirBufferC1ERNS_9AllocatorE(%"class.nanojit::LirBuffer"*, %"class.nanojit::Allocator"*)

declare hidden void @_ZN2js14AbortProfilingEP9JSContext(%struct.JSContext* nocapture) nounwind

declare void @_ZN2js4tjit6Writer4initEPN7nanojit10LogControlE(%"class.js::tjit::Writer"*, %"class.nanojit::LogControl"*)

declare hidden void @_ZN2js13TraceRecorder6importEPNS_12TreeFragmentEPN7nanojit4LInsEjjjP11JSValueType(%"class.js::TraceRecorder"*, %"struct.js::TreeFragment"* nocapture, %"class.nanojit::LIns"*, i32, i32, i32, i8*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder5guardEbPN7nanojit4LInsENS_8ExitTypeEb(%"class.js::TraceRecorder"*, i1 zeroext, %"class.nanojit::LIns"*, i32, i1 zeroext) nounwind align 2

declare hidden void @_ZN2js13TraceRecorderD1Ev(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorderD2Ev(%"class.js::TraceRecorder"*) nounwind align 2

declare fastcc void @_ZN2js9TrashTreeEPNS_12TreeFragmentE(%"struct.js::TreeFragment"* nocapture) nounwind

declare void @_ZN7nanojit9Allocator5resetEv(%"class.nanojit::Allocator"*)

declare hidden void @_ZN2js13TraceRecorder19forgetGuardedShapesEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18finishSuccessfullyEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden zeroext i1 @_ZN2js16OverfullJITCacheEP9JSContextPNS_12TraceMonitorE(%struct.JSContext* nocapture, %"struct.js::TraceMonitor"* nocapture) nounwind

declare hidden i32 @_ZN2js13TraceRecorder11finishAbortEPKc(%"class.js::TraceRecorder"*, i8* nocapture) nounwind align 2

declare hidden void @_ZN2js5QueueIPNS_10VMSideExitEE9setLengthEj(%17* nocapture, i32) nounwind align 2

declare hidden i32 @_ZNK2js13TraceRecorder16nativeGlobalSlotEPKNS_5ValueE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"*) nounwind readonly align 2

declare hidden i32 @_ZNK2js13TraceRecorder18nativeGlobalOffsetEPKNS_5ValueE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"*) nounwind readonly align 2

declare hidden zeroext i1 @_ZNK2js13TraceRecorder8isGlobalEPKNS_5ValueE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"*) nounwind readonly align 2

declare hidden zeroext i1 @_ZNK2js13TraceRecorder15isVoidPtrGlobalEPKv(%"class.js::TraceRecorder"* nocapture, i8*) nounwind readonly align 2

declare hidden i32 @_ZNK2js13TraceRecorder21nativeStackOffsetImplEPKv(%"class.js::TraceRecorder"* nocapture, i8*) nounwind align 2

declare hidden void @_ZN2js12TraceMonitor5flushEv(%"struct.js::TraceMonitor"* nocapture) nounwind align 2

declare void @_ZN2js4mjit14ResetTraceHintEP8JSScriptPhtb(%struct.JSScript*, i8*, i16 zeroext, i1 zeroext)

declare void @_ZN7nanojit9CodeAlloc5resetEv(%"class.nanojit::CodeAlloc"*)

declare void @_ZN7nanojit9AssemblerC1ERNS_9CodeAllocERNS_9AllocatorES4_PN7avmplus7AvmCoreEPNS_10LogControlERKNS_6ConfigE(%"class.nanojit::Assembler"*, %"class.nanojit::CodeAlloc"*, %"class.nanojit::Allocator"*, %"class.nanojit::Allocator"*, %"class.avmplus::AvmCore"*, %"class.nanojit::LogControl"*, %"struct.nanojit::Config"*)

declare hidden void @_ZN2js12TraceMonitor5sweepEP9JSContext(%"struct.js::TraceMonitor"* nocapture, %struct.JSContext*) nounwind align 2

declare hidden void @_ZN2js12TraceMonitor4markEP8JSTracer(%"struct.js::TraceMonitor"* nocapture, %struct.JSTracer*) nounwind align 2

declare hidden void @_ZN2js19ExternNativeToValueEP9JSContextRNS_5ValueE11JSValueTypePd(%struct.JSContext* nocapture, %"class.js::Value"* nocapture, i8 zeroext, double* nocapture) nounwind

declare hidden x86_fastcallcc i32 @_ZN2js18GetUpvarArgOnTraceEP9JSContextjijPd(%struct.JSContext* nocapture, i32, i32, i32, double* nocapture) nounwind

declare hidden zeroext i8 @_ZN2js15GetUpvarOnTraceINS_14UpvarArgTraitsEEE11JSValueTypeP9JSContextjijPd(%struct.JSContext* nocapture, i32, i32, i32, double* nocapture) nounwind inlinehint

declare hidden x86_fastcallcc i32 @_ZN2js18GetUpvarVarOnTraceEP9JSContextjijPd(%struct.JSContext* nocapture, i32, i32, i32, double* nocapture) nounwind

declare hidden zeroext i8 @_ZN2js15GetUpvarOnTraceINS_14UpvarVarTraitsEEE11JSValueTypeP9JSContextjijPd(%struct.JSContext* nocapture, i32, i32, i32, double* nocapture) nounwind inlinehint

declare hidden x86_fastcallcc i32 @_ZN2js20GetUpvarStackOnTraceEP9JSContextjijPd(%struct.JSContext* nocapture, i32, i32, i32, double* nocapture) nounwind

declare hidden zeroext i8 @_ZN2js15GetUpvarOnTraceINS_16UpvarStackTraitsEEE11JSValueTypeP9JSContextjijPd(%struct.JSContext* nocapture, i32, i32, i32, double* nocapture) nounwind inlinehint

declare hidden x86_fastcallcc i32 @_ZN2js13GetClosureArgEP9JSContextP8JSObjectPKNS_14ClosureVarInfoEPd(%struct.JSContext* nocapture, %struct.JSObject* nocapture, %"class.js::KidsPointer"* nocapture, double* nocapture) nounwind

declare hidden x86_fastcallcc i32 @_ZN2js13GetClosureVarEP9JSContextP8JSObjectPKNS_14ClosureVarInfoEPd(%struct.JSContext* nocapture, %struct.JSObject* nocapture, %"class.js::KidsPointer"* nocapture, double* nocapture) nounwind

declare hidden void @_ZN2js13TraceRecorder10importImplENS_4tjit7AddressEPKv11JSValueTypePKcjP12JSStackFrame(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval, i8*, i8 zeroext, i8* nocapture, i32, %struct.JSStackFrame* nocapture) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder6importENS_4tjit7AddressEPKNS_5ValueE11JSValueTypePKcjP12JSStackFrame(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval, %"class.js::Value"*, i8 zeroext, i8* nocapture, i32, %struct.JSStackFrame* nocapture) nounwind align 2

declare hidden zeroext i1 @_ZN2js13TraceRecorder11isValidSlotEP8JSObjectPKNS_5ShapeE(%"class.js::TraceRecorder"* nocapture, %struct.JSObject* nocapture, %"struct.js::Shape"* nocapture) nounwind readonly align 2

declare hidden void @_ZN2js13TraceRecorder16importGlobalSlotEj(%"class.js::TraceRecorder"* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js5QueueItE3addEt(%38* nocapture, i16 zeroext) nounwind align 2

declare hidden void @_ZN2js5QueueI11JSValueTypeE3addES1_(%78* nocapture, i8 zeroext) nounwind align 2

declare hidden zeroext i1 @_ZN2js13TraceRecorder22lazilyImportGlobalSlotEj(%"class.js::TraceRecorder"* nocapture, i32) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder9writeBackEPN7nanojit4LInsES3_ib(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, i32, i1 zeroext) nounwind align 2

declare zeroext i1 @_ZN2js4tjit15IsPromotedInt32EPN7nanojit4LInsE(%"class.nanojit::LIns"*)

declare hidden void @_ZN2js13TraceRecorder7setImplEPvPN7nanojit4LInsEb(%"class.js::TraceRecorder"*, i8*, %"class.nanojit::LIns"*, i1 zeroext) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder14setFrameObjPtrEPvPN7nanojit4LInsEb(%"class.js::TraceRecorder"*, i8*, %"class.nanojit::LIns"*, i1 zeroext) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder13attemptImportEPKNS_5ValueE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder3getEPKNS_5ValueE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder7getImplEPKv(%"class.js::TraceRecorder"* nocapture, i8*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder14getFrameObjPtrEPv(%"class.js::TraceRecorder"* nocapture, i8*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder4addrEPNS_5ValueE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder38checkForGlobalObjectReallocationHelperEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder17adjustCallerTypesEPNS_12TreeFragmentE(%"class.js::TraceRecorder"*, %"struct.js::TreeFragment"* nocapture) nounwind align 2

declare hidden %"struct.js::VMSideExit"* @_ZN2js13TraceRecorder8snapshotENS_8ExitTypeE(%"class.js::TraceRecorder"*, i32) nounwind align 2

declare i32 @_Z13js_InferFlagsP9JSContextj(%struct.JSContext*, i32)

declare hidden %"struct.nanojit::GuardRecord"* @_ZN2js13TraceRecorder17createGuardRecordEPNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder10ensureCondEPPN7nanojit4LInsEPb(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"** nocapture, i8* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder5guardEbPN7nanojit4LInsEPNS_10VMSideExitEb(%"class.js::TraceRecorder"* nocapture, i1 zeroext, %"class.nanojit::LIns"*, %"struct.js::VMSideExit"*, i1 zeroext) nounwind align 2

declare hidden void @_ZN2js5QueueIPNS_10VMSideExitEE3addES2_(%17* nocapture, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder9guard_xovEN7nanojit7LOpcodeEPNS1_4LInsES4_PNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, i32, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden %"struct.js::VMSideExit"* @_ZN2js13TraceRecorder4copyEPNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %"struct.js::VMSideExit"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7compileEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare void @_ZN7nanojit9Assembler7compileEPNS_8FragmentERNS_9AllocatorEb(%"class.nanojit::Assembler"*, %"class.nanojit::Fragment"*, %"class.nanojit::Allocator"*, i1 zeroext)

declare void @_ZN7nanojit9Assembler5patchEPNS_8SideExitEPNS_10SwitchInfoE(%"class.nanojit::Assembler"*, %"struct.nanojit::SideExit"*, %"struct.nanojit::SwitchInfo"*)

declare void @_ZN7nanojit9Assembler5patchEPNS_8SideExitE(%"class.nanojit::Assembler"*, %"struct.nanojit::SideExit"*)

declare hidden i32 @_ZN2js13TraceRecorder17selfTypeStabilityERNS_7SlotMapE(%"class.js::TraceRecorder"* nocapture, %"class.js::SlotMap"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17peerTypeStabilityERNS_7SlotMapEPKvPPNS_12TreeFragmentE(%"class.js::TraceRecorder"* nocapture, %"class.js::SlotMap"* nocapture, i8* nocapture, %"struct.js::TreeFragment"** nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder9closeLoopEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder16joinEdgesToEntryEPNS_12TreeFragmentE(%"class.js::TraceRecorder"* nocapture, %"struct.js::TreeFragment"*) nounwind align 2

declare fastcc void @_ZN2js18AttemptCompilationEPNS_12TraceMonitorEP8JSObjectP8JSScriptPhj(%"struct.js::TraceMonitor"* nocapture, %struct.JSObject*, %struct.JSScript*, i8*, i32) nounwind

declare hidden void @_ZN2js14DefaultSlotMapD1Ev(%"class.js::DefaultSlotMap"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23findUndemotesInTypemapsERKNS_7TypeMapEPNS_16LinkableFragmentERNS_5QueueIjEE(%"class.js::TraceRecorder"* nocapture, %"class.js::TypeMap"* nocapture, %"struct.js::LinkableFragment"* nocapture, %79* nocapture) nounwind align 2

declare hidden void @_ZN2js5QueueIjE9setLengthEj(%79* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js5QueueIjE3addEj(%79* nocapture, i32) nounwind align 2

declare fastcc void @_ZN2js15FullMapFromExitERNS_7TypeMapEPNS_10VMSideExitE(%"class.js::TypeMap"* nocapture, %"struct.js::VMSideExit"* nocapture) nounwind

declare fastcc i32 @_ZN2js18TypeMapLinkabilityEP9JSContextPNS_12TraceMonitorERKNS_7TypeMapEPNS_12TreeFragmentE(%struct.JSContext* nocapture, %"struct.js::TraceMonitor"* nocapture, %"class.js::TypeMap"* nocapture, %"struct.js::TreeFragment"* nocapture) nounwind

declare fastcc void @_ZN2js28SpecializeTreesToLateGlobalsEP9JSContextPNS_12TreeFragmentEP11JSValueTypej(%"struct.js::TreeFragment"* nocapture, i8*, i32) nounwind

declare hidden i32 @_ZN2js13TraceRecorder7endLoopEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7endLoopEPNS_10VMSideExitE(%"class.js::TraceRecorder"*, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder15prepareTreeCallEPNS_12TreeFragmentE(%"class.js::TraceRecorder"*, %"struct.js::TreeFragment"* nocapture) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder12emitTreeCallEPNS_12TreeFragmentEPNS_10VMSideExitE(%"class.js::TraceRecorder"*, %"struct.js::TreeFragment"*, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder14trackCfgMergesEPh(%"class.js::TraceRecorder"* nocapture, i8*) nounwind align 2

declare i8* @_Z19js_GetSrcNoteCachedP9JSContextP8JSScriptPh(%struct.JSContext*, %struct.JSScript*, i8*)

declare hidden void @_ZN2js5QueueIPhE3addES1_(%31* nocapture, i8*) nounwind align 2

declare i32 @js_GetSrcNoteOffset(i8*, i32)

declare hidden void @_ZN2js13TraceRecorder6emitIfEPhbPN7nanojit4LInsE(%"class.js::TraceRecorder"*, i8*, i1 zeroext, %"class.nanojit::LIns"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder6fuseIfEPhbPN7nanojit4LInsE(%"class.js::TraceRecorder"*, i8*, i1 zeroext, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder13checkTraceEndEPh(%"class.js::TraceRecorder"*, i8*) nounwind align 2

declare hidden zeroext i1 @_ZN2js13TraceRecorder13startRecorderEP9JSContextPNS_12TraceMonitorEPNS_10VMSideExitEPNS_10VMFragmentEjjP11JSValueTypeS6_P8JSScriptPhjb(%struct.JSContext*, %"struct.js::TraceMonitor"*, %"struct.js::VMSideExit"*, %"class.js::VMFragment"*, i32, i32, i8*, %"struct.js::VMSideExit"*, %struct.JSScript*, i8*, i32, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14recordLoopEdgeEP9JSContextPS0_Rj(%struct.JSContext*, %"class.js::TraceRecorder"*, i32*) nounwind align 2

declare fastcc %"struct.js::TreeFragment"* @_ZN2js15LookupOrAddLoopEPNS_12TraceMonitorEPKvP8JSObjectjj(%"struct.js::TraceMonitor"*, i8*, %struct.JSObject*, i32, i32) nounwind

declare %struct.JSObject* @_ZNK8JSObject9getGlobalEv(%struct.JSObject*)

declare fastcc zeroext i1 @_ZN2js22CheckGlobalObjectShapeEP9JSContextPNS_12TraceMonitorEP8JSObjectPjPPNS_5QueueItEE(%struct.JSContext*, %"struct.js::TraceMonitor"*, %struct.JSObject*, i32*, %38**) nounwind

declare hidden %"struct.js::TreeFragment"* @_ZN2js13TraceRecorder24findNestedCompatiblePeerEPNS_12TreeFragmentE(%"class.js::TraceRecorder"*, %"struct.js::TreeFragment"*) nounwind align 2

declare hidden i32 @_ZN2js18AbortRecordingImplEP9JSContext(%struct.JSContext* nocapture) nounwind

declare fastcc zeroext i1 @_ZN2js10RecordTreeEP9JSContextPNS_12TraceMonitorEPNS_12TreeFragmentEP8JSScriptPhjPNS_5QueueItEE(%struct.JSContext*, %"struct.js::TraceMonitor"*, %"struct.js::TreeFragment"*, %struct.JSScript*, i8*, i32, %38*) nounwind

declare hidden i32 @_ZN2js13TraceRecorder15attemptTreeCallEPNS_12TreeFragmentERj(%"class.js::TraceRecorder"*, %"struct.js::TreeFragment"*, i32*) nounwind align 2

declare fastcc zeroext i1 @_ZN2js11ExecuteTreeEP9JSContextPNS_12TraceMonitorEPNS_12TreeFragmentERjPPNS_10VMSideExitES9_(%struct.JSContext*, %"struct.js::TraceMonitor"*, %"struct.js::TreeFragment"*, i32*, %"struct.js::VMSideExit"**, %"struct.js::VMSideExit"** nocapture) nounwind

declare fastcc zeroext i1 @_ZN2js19AttemptToExtendTreeEP9JSContextPNS_12TraceMonitorEPNS_10VMSideExitES5_P8JSScriptPh(%struct.JSContext*, %"struct.js::TraceMonitor"*, %"struct.js::VMSideExit"*, %"struct.js::VMSideExit"*, %struct.JSScript*, i8*) nounwind

declare fastcc zeroext i1 @_ZN2js22AttemptToStabilizeTreeEP9JSContextPNS_12TraceMonitorEP8JSObjectPNS_10VMSideExitEP8JSScriptPhj(%struct.JSContext*, %"struct.js::TraceMonitor"*, %struct.JSObject* nocapture, %"struct.js::VMSideExit"*, %struct.JSScript*, i8*, i32) nounwind

declare hidden i32 @_ZN2js14RecordLoopEdgeEP9JSContextPNS_12TraceMonitorERj(%struct.JSContext*, %"struct.js::TraceMonitor"*, i32*) nounwind

declare fastcc %"struct.js::TreeFragment"* @_ZN2js20FindVMCompatiblePeerEP9JSContextP8JSObjectPNS_12TreeFragmentERj(%struct.JSContext*, %struct.JSObject* nocapture, %"struct.js::TreeFragment"*, i32* nocapture) nounwind

declare hidden i32 @_ZN2js13TraceRecorder16monitorRecordingE4JSOp(%"class.js::TraceRecorder"*, i32) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder11unbox_valueERKNS_5ValueENS_4tjit7AddressEPNS_10VMSideExitEb(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"* nocapture, %"struct.js::tjit::Address"* nocapture byval, %"struct.js::VMSideExit"*, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_NOPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_PUSHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_POPVEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_ENTERWITHEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_LEAVEWITHEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_RETURNEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_GOTOEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_IFEQEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_IFNEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_ARGUMENTSEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_FORARGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_FORLOCALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_DUPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_DUP2Ev(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_SETCONSTEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_BITOREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_BITXOREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_BITANDEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_EQEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_NEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_LTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_LEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_GTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_GEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_LSHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_RSHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_URSHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_ADDEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_SUBEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_MULEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_DIVEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_MODEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_NOTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_BITNOTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_NEGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_POSEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DELNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DELPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DELELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_TYPEOFEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_VOIDEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_INCNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_INCPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_INCELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DECNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DECPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DECELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_NAMEINCEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_PROPINCEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_ELEMINCEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_NAMEDECEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_PROPDECEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_ELEMDECEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_GETPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_SETPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_GETELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_SETELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_CALLNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_CALLEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_NAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_DOUBLEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_STRINGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_ZEROEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_ONEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_NULLEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_THISEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_FALSEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_TRUEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_OREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_ANDEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_TABLESWITCHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder24record_JSOP_LOOKUPSWITCHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_STRICTEQEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_STRICTNEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_SETCALLEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_ITEREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_MOREITEREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_ENDITEREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_FUNAPPLYEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_SWAPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_OBJECTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_POPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_NEWEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_TRAPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_GETARGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_SETARGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_GETLOCALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_SETLOCALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_UINT16Ev(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_NEWINITEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_NEWARRAYEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_NEWOBJECTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_ENDINITEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_INITPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_INITELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_DEFSHARPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_USESHARPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_INCARGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_DECARGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_ARGINCEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_ARGDECEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_INCLOCALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_DECLOCALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_LOCALINCEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_LOCALDECEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_IMACOPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_FORNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_FORPROPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_FORELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_POPNEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_BINDNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_SETNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_THROWEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder14record_JSOP_INEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_INSTANCEOFEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_DEBUGGEREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_GOSUBEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_RETSUBEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_EXCEPTIONEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_LINENOEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_CONDSWITCHEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_CASEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DEFAULTEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_EVALEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_ENUMELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_GETTEREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_SETTEREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_DEFFUNEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_DEFCONSTEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_DEFVAREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_LAMBDAEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_CALLEEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_SETLOCALPOPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_PICKEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_TRYEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_FINALLYEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_GETFCSLOTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_CALLFCSLOTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_ARGSUBEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_ARGCNTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_DEFLOCALFUNEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_GOTOXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_IFEQXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_IFNEXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15record_JSOP_ORXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_ANDXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_GOSUBXEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_CASEXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_DEFAULTXEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder24record_JSOP_TABLESWITCHXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder25record_JSOP_LOOKUPSWITCHXEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_BACKPATCHEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder25record_JSOP_BACKPATCH_POPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_THROWINGEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_SETRVALEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_RETRVALEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_GETGNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_SETGNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_INCGNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_DECGNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_GNAMEINCEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_GNAMEDECEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_REGEXPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_DEFXMLNSEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_ANYNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_QNAMEPARTEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_QNAMECONSTEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_QNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_TOATTRNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_TOATTRVALEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_ADDATTRNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_ADDATTRVALEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_BINDXMLNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_SETXMLNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_XMLNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_DESCENDANTSEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_FILTEREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_ENDFILTEREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_TOXMLEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_TOXMLLISTEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_XMLTAGEXPREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_XMLELTEXPREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_NOTRACEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_XMLCDATAEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_XMLCOMMENTEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_XMLPIEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_DELDESCEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_CALLPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_BLOCKCHAINEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder26record_JSOP_NULLBLOCKCHAINEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_UINT24Ev(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_INDEXBASEEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_RESETBASEEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_RESETBASE0Ev(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_STARTXMLEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder24record_JSOP_STARTXMLEXPREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_CALLELEMEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_STOPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_GETXPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_CALLXMLNAMEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_TYPEOFEXPREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_ENTERBLOCKEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_LEAVEBLOCKEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_IFPRIMTOPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_PRIMTOPEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_GENERATOREv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_YIELDEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_ARRAYPUSHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_GETFUNNSEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder25record_JSOP_ENUMCONSTELEMEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder26record_JSOP_LEAVEBLOCKEXPREv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_GETTHISPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_GETARGPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder24record_JSOP_GETLOCALPROPEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_INDEXBASE1Ev(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_INDEXBASE2Ev(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_INDEXBASE3Ev(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_CALLGNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_CALLLOCALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_CALLARGEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_BINDGNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_INT8Ev(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_INT32Ev(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_LENGTHEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16record_JSOP_HOLEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_DEFFUN_FCEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder26record_JSOP_DEFLOCALFUN_FCEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_LAMBDA_FCEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder18record_JSOP_OBJTOPEv(%"class.js::TraceRecorder"* nocapture) nounwind readonly align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_JSOP_TRACEEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder24record_JSOP_GETUPVAR_DBGEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder25record_JSOP_CALLUPVAR_DBGEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder24record_JSOP_DEFFUN_DBGFCEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder29record_JSOP_DEFLOCALFUN_DBGFCEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder24record_JSOP_LAMBDA_DBGFCEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_SETMETHODEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_INITMETHODEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_UNBRANDEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23record_JSOP_UNBRANDTHISEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_SHARPINITEv(%"class.js::TraceRecorder"* nocapture) nounwind readnone align 2

declare hidden i32 @_ZN2js13TraceRecorder21record_JSOP_GETGLOBALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22record_JSOP_CALLGLOBALEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19record_JSOP_FUNCALLEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20record_JSOP_FORGNAMEEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden void @_ZN2js20SetMaxCodeCacheBytesEP9JSContextj(%struct.JSContext* nocapture, i32) nounwind

declare hidden zeroext i1 @_ZN2js7InitJITEPNS_12TraceMonitorE(%"struct.js::TraceMonitor"* nocapture) nounwind

declare hidden void @_ZN2js9FinishJITEPNS_12TraceMonitorE(%"struct.js::TraceMonitor"* nocapture) nounwind

declare hidden void @_ZN2js20PurgeScriptFragmentsEPNS_12TraceMonitorEP8JSScript(%"struct.js::TraceMonitor"* nocapture, %struct.JSScript*) nounwind

declare i32 @_ZN7nanojit9CodeAlloc4sizeEv(%"class.nanojit::CodeAlloc"*)

declare void @_ZN2js8DeepBailEP9JSContext(%struct.JSContext* nocapture) nounwind

declare fastcc void @_ZN2js9LeaveTreeEPNS_12TraceMonitorERNS_11TracerStateEPNS_10VMSideExitE(%"struct.js::TraceMonitor"* nocapture, %"struct.js::TracerState"* nocapture, %"struct.js::VMSideExit"*) nounwind

declare hidden %"class.js::Value"* @_ZNK2js13TraceRecorder6argvalEj(%"class.js::TraceRecorder"* nocapture, i32) nounwind readonly align 2

declare hidden %"class.js::Value"* @_ZNK2js13TraceRecorder6varvalEj(%"class.js::TraceRecorder"* nocapture, i32) nounwind readonly align 2

declare hidden %"class.js::Value"* @_ZNK2js13TraceRecorder8stackvalEi(%"class.js::TraceRecorder"* nocapture, i32) nounwind readonly align 2

declare hidden void @_ZN2js13TraceRecorder11updateAtomsEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder11updateAtomsEP8JSScript(%"class.js::TraceRecorder"* nocapture, %struct.JSScript* nocapture) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder10scopeChainEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZNK2js13TraceRecorder15entryScopeChainEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZNK2js13TraceRecorder13entryFrameInsEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden %struct.JSStackFrame* @_ZNK2js13TraceRecorder14frameIfInRangeEP8JSObjectPj(%"class.js::TraceRecorder"* nocapture, %struct.JSObject* nocapture, i32*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14scopeChainPropEP8JSObjectRPNS_5ValueERPN7nanojit4LInsERNS0_10NameResultE(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.js::Value"** nocapture, %"class.nanojit::LIns"** nocapture, %"struct.js::TraceRecorder::NameResult"*) nounwind align 2

declare i32 @_Z15js_FindPropertyP9JSContextiPP8JSObjectS3_PP10JSProperty(%struct.JSContext*, i32, %struct.JSObject**, %struct.JSObject**, %struct.JSProperty**)

declare hidden i32 @_ZN2js13TraceRecorder18traverseScopeChainEP8JSObjectPN7nanojit4LInsES2_RS5_(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, %struct.JSObject*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder8callPropEP8JSObjectP10JSPropertyiRPNS_5ValueERPN7nanojit4LInsERNS0_10NameResultE(%"class.js::TraceRecorder"*, %struct.JSObject*, %struct.JSProperty*, i32, %"class.js::Value"** nocapture, %"class.nanojit::LIns"** nocapture, %"struct.js::TraceRecorder::NameResult"*) nounwind align 2

declare i32 @_ZN2js10GetCallArgEP9JSContextP8JSObjectiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)

declare i32 @_ZN2js10GetCallVarEP9JSContextP8JSObjectiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)

declare i32 @_ZN2js17GetCallVarCheckedEP9JSContextP8JSObjectiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)

declare i32 @_Z20js_GetPropertyHelperP9JSContextP8JSObjectijPN2js5ValueE(%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder10unbox_slotEP8JSObjectPN7nanojit4LInsEjPNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %struct.JSObject*, %"class.nanojit::LIns"*, i32, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder9stackLoadENS_4tjit7AddressEh(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval, i8 zeroext) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder3argEj(%"class.js::TraceRecorder"* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder3argEjPN7nanojit4LInsE(%"class.js::TraceRecorder"*, i32, %"class.nanojit::LIns"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder3varEj(%"class.js::TraceRecorder"* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder3varEjPN7nanojit4LInsE(%"class.js::TraceRecorder"*, i32, %"class.nanojit::LIns"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder5stackEi(%"class.js::TraceRecorder"* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder5stackEiPN7nanojit4LInsE(%"class.js::TraceRecorder"*, i32, %"class.nanojit::LIns"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder11guardNonNegEPN7nanojit4LInsES3_PNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder3aluEN7nanojit7LOpcodeEddPNS1_4LInsES4_(%"class.js::TraceRecorder"*, i32, double, double, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*) nounwind align 2

declare x86_fastcallcc double @_Z7js_dmoddd(double, double)

declare i32 @_ZN7nanojit14arithOpcodeD2IENS_7LOpcodeE(i32)

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder3d2iEPN7nanojit4LInsEb(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, i1 zeroext) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder3d2uEPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare i32 @_Z21js_DoubleToECMAUint32d(double)

declare hidden i32 @_ZN2js13TraceRecorder15makeNumberInt32EPN7nanojit4LInsEPS3_(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder16makeNumberUint32EPN7nanojit4LInsEPS3_(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare zeroext i1 @_ZN2js4tjit16IsPromotedUint32EPN7nanojit4LInsE(%"class.nanojit::LIns"*)

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder9stringifyERKNS_5ValueE(%"class.js::TraceRecorder"*, %"class.js::Value"*) nounwind align 2

declare hidden zeroext i1 @_ZNK2js13TraceRecorder13canCallImacroEv(%"class.js::TraceRecorder"* nocapture) nounwind readonly align 2

declare hidden i32 @_ZN2js13TraceRecorder10callImacroEPh(%"class.js::TraceRecorder"* nocapture, i8*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20callImacroInfalliblyEPh(%"class.js::TraceRecorder"* nocapture, i8*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder4ifopEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder11tableswitchEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder8switchopEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder3incERNS_5ValueEib(%"class.js::TraceRecorder"*, %"class.js::Value"*, i32, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder3incERKNS_5ValueERPN7nanojit4LInsERS1_ib(%"class.js::TraceRecorder"*, %"class.js::Value"* nocapture, %"class.nanojit::LIns"** nocapture, %"class.js::Value"* nocapture, i32, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder9incHelperERKNS_5ValueERPN7nanojit4LInsERS1_S7_i(%"class.js::TraceRecorder"*, %"class.js::Value"* nocapture, %"class.nanojit::LIns"** nocapture, %"class.js::Value"* nocapture, %"class.nanojit::LIns"** nocapture, i32) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7incPropEib(%"class.js::TraceRecorder"*, i32, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder4propEP8JSObjectPN7nanojit4LInsEPjPS5_PNS_5ValueE(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, i32*, %"class.nanojit::LIns"** nocapture, %"class.js::Value"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder14stobj_set_slotEP8JSObjectPN7nanojit4LInsEjRS5_RKNS_5ValueES5_(%"class.js::TraceRecorder"* nocapture, %struct.JSObject*, %"class.nanojit::LIns"*, i32, %"class.nanojit::LIns"** nocapture, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7incElemEib(%"class.js::TraceRecorder"*, i32, i1 zeroext) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder15guardDenseArrayEPN7nanojit4LInsENS_8ExitTypeE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, i32) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17denseArrayElementERNS_5ValueES2_RPS1_RPN7nanojit4LInsES8_PNS_10VMSideExitE(%"class.js::TraceRecorder"*, %"class.js::Value"*, %"class.js::Value"*, %"class.js::Value"** nocapture, %"class.nanojit::LIns"** nocapture, %"class.nanojit::LIns"** nocapture, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder14box_value_intoERKNS_5ValueEPN7nanojit4LInsENS_4tjit7AddressE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*, %"struct.js::tjit::Address"* nocapture byval) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14strictEqualityEbb(%"class.js::TraceRecorder"*, i1 zeroext, i1 zeroext) nounwind align 2

declare zeroext i1 @_ZN2js12EqualStringsEP9JSContextP8JSStringS3_Pi(%struct.JSContext*, %struct.JSString*, %struct.JSString*, i32*)

declare hidden i32 @_ZN2js13TraceRecorder8equalityEbb(%"class.js::TraceRecorder"*, i1 zeroext, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14equalityHelperERNS_5ValueES2_PN7nanojit4LInsES5_bbS2_(%"class.js::TraceRecorder"*, %"class.js::Value"*, %"class.js::Value"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, i1 zeroext, i1 zeroext, %"class.js::Value"*) nounwind align 2

declare x86_fastcallcc double @_Z17js_StringToNumberP9JSContextP8JSStringPi(%struct.JSContext*, %struct.JSString*, i32*)

declare hidden i32 @_ZN2js13TraceRecorder21guardNativeConversionERNS_5ValueE(%"class.js::TraceRecorder"*, %"class.js::Value"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder10relationalEN7nanojit7LOpcodeEb(%"class.js::TraceRecorder"*, i32, i1 zeroext) nounwind align 2

declare i32 @_ZN7nanojit12cmpOpcodeD2IENS_7LOpcodeE(i32)

declare hidden i32 @_ZN2js13TraceRecorder5unaryEN7nanojit7LOpcodeE(%"class.js::TraceRecorder"*, i32) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder6binaryEN7nanojit7LOpcodeE(%"class.js::TraceRecorder"*, i32) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder10guardShapeEPN7nanojit4LInsEP8JSObjectjPKcPNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, %struct.JSObject*, i32, i8* nocapture, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder28forgetGuardedShapesForObjectEP8JSObject(%"class.js::TraceRecorder"*, %struct.JSObject*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19test_property_cacheEP8JSObjectPN7nanojit4LInsERS2_RNS_5PCValE(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, %struct.JSObject**, %"class.js::KidsPointer"* nocapture) nounwind align 2

declare %"struct.js::PropertyCacheEntry"* @_Z21js_FindPropertyHelperP9JSContextiiPP8JSObjectS3_PP10JSProperty(%struct.JSContext*, i32, i32, %struct.JSObject**, %struct.JSObject**, %struct.JSProperty**)

declare i32 @_Z26js_LookupPropertyWithFlagsP9JSContextP8JSObjectijPS2_PP10JSProperty(%struct.JSContext*, %struct.JSObject*, i32, i32, %struct.JSObject**, %struct.JSProperty**)

declare %"struct.js::PropertyCacheEntry"* @_ZN2js13PropertyCache4fillEP9JSContextP8JSObjectjjS4_PKNS_5ShapeEi(%"class.js::PropertyCache"*, %struct.JSContext*, %struct.JSObject*, i32, i32, %struct.JSObject*, %"struct.js::Shape"*, i32)

declare hidden i32 @_ZN2js13TraceRecorder21guardPropertyCacheHitEPN7nanojit4LInsEP8JSObjectS5_PNS_18PropertyCacheEntryERNS_5PCValE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %struct.JSObject*, %struct.JSObject*, %"struct.js::PropertyCacheEntry"* nocapture, %"class.js::KidsPointer"* nocapture) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder15stobj_set_fslotEPN7nanojit4LInsEjRKNS_5ValueES3_(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, i32, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder15stobj_set_dslotEPN7nanojit4LInsEjRS3_RKNS_5ValueES3_(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, i32, %"class.nanojit::LIns"** nocapture, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder18box_undefined_intoENS_4tjit7AddressE(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder13box_null_intoENS_4tjit7AddressE(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder12unbox_objectENS_4tjit7AddressEPN7nanojit4LInsE11JSValueTypePNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval, %"class.nanojit::LIns"*, i8 zeroext, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder10guardClassEPN7nanojit4LInsEPNS_5ClassEPNS_10VMSideExitENS1_8LoadQualE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, %"struct.js::Class"*, %"struct.js::VMSideExit"*, i32) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder13guardNotClassEPN7nanojit4LInsEPNS_5ClassEPNS_10VMSideExitENS1_8LoadQualE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, %"struct.js::Class"*, %"struct.js::VMSideExit"*, i32) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder23unbox_non_double_objectENS_4tjit7AddressEPN7nanojit4LInsE11JSValueTypePNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval, %"class.nanojit::LIns"*, i8 zeroext, %"struct.js::VMSideExit"*) nounwind inlinehint align 2

declare hidden void @_ZN2js13TraceRecorder16unbox_any_objectENS_4tjit7AddressEPPN7nanojit4LInsES6_(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval, %"class.nanojit::LIns"** nocapture, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder13is_boxed_trueENS_4tjit7AddressE(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder14is_boxed_magicENS_4tjit7AddressE10JSWhyMagic(%"class.js::TraceRecorder"* nocapture, %"struct.js::tjit::Address"* nocapture byval, i32) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder25box_value_for_native_callERKNS_5ValueEPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder20box_value_into_allocERKNS_5ValueEPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder12is_string_idEPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder15unbox_string_idEPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*) nounwind readnone align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder12unbox_int_idEPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7getThisERPN7nanojit4LInsE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder16guardClassHelperEbPN7nanojit4LInsEPNS_5ClassEPNS_10VMSideExitENS1_8LoadQualE(%"class.js::TraceRecorder"* nocapture, i1 zeroext, %"class.nanojit::LIns"*, %"struct.js::Class"*, %"struct.js::VMSideExit"*, i32) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder15guardDenseArrayEPN7nanojit4LInsEPNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden zeroext i1 @_ZN2js13TraceRecorder17guardHasPrototypeEP8JSObjectPN7nanojit4LInsEPS2_PS5_PNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %struct.JSObject* nocapture, %"class.nanojit::LIns"*, %struct.JSObject** nocapture, %"class.nanojit::LIns"** nocapture, %"struct.js::VMSideExit"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder36guardPrototypeHasNoIndexedPropertiesEP8JSObjectPN7nanojit4LInsEPNS_10VMSideExitE(%"class.js::TraceRecorder"* nocapture, %struct.JSObject*, %"class.nanojit::LIns"* nocapture, %"struct.js::VMSideExit"*) nounwind align 2

declare i32 @_Z32js_PrototypeHasIndexedPropertiesP9JSContextP8JSObject(%struct.JSContext*, %struct.JSObject*)

declare i32 @JS_ConvertStub(%struct.JSContext*, %struct.JSObject*, i32, i64*)

declare i32 @_Z13js_TryValueOfP9JSContextP8JSObject6JSTypePN2js5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)

declare hidden void @_ZN2js13TraceRecorder36clearReturningFrameFromNativeTrackerEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20putActivationObjectsEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden void @_ZN12JSStackFrame25forEachCanonicalActualArgIN2js6BoxArgEEEvT_(%struct.JSStackFrame*, %"class.js::BoxArg"* nocapture byval) nounwind inlinehint align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_EnterFrameEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17record_LeaveFrameEv(%"class.js::TraceRecorder"*) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js13functionProbeEP9JSContextP10JSFunctioni(%struct.JSContext* nocapture, %struct.JSFunction* nocapture, i32) nounwind readnone

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder12newArgumentsEPN7nanojit4LInsE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17getClassPrototypeEP8JSObjectRPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %struct.JSObject*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder17getClassPrototypeE10JSProtoKeyRPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, i32, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare i32 @_Z20js_GetClassPrototypeP9JSContextP8JSObject10JSProtoKeyPS2_PN2js5ClassE(%struct.JSContext*, %struct.JSObject*, i32, %struct.JSObject**, %"struct.js::Class"*)

declare hidden i32 @_ZN2js13TraceRecorder9newStringEP8JSObjectjPNS_5ValueES4_(%"class.js::TraceRecorder"*, %struct.JSObject*, i32, %"class.js::Value"*, %"class.js::Value"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder8newArrayEP8JSObjectjPNS_5ValueES4_(%"class.js::TraceRecorder"*, %struct.JSObject*, i32, %"class.js::Value"*, %"class.js::Value"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder31propagateFailureToBuiltinStatusEPN7nanojit4LInsERS3_(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder20emitNativePropertyOpEPKNS_5ShapeEPN7nanojit4LInsEbS6_(%"class.js::TraceRecorder"*, %"struct.js::Shape"* nocapture, %"class.nanojit::LIns"*, i1 zeroext, %"class.nanojit::LIns"*) nounwind align 2

declare hidden %"struct.js::VMSideExit"* @_ZN2js13TraceRecorder17enterDeepBailCallEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder17leaveDeepBailCallEv(%"class.js::TraceRecorder"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14emitNativeCallEP19JSSpecializedNativejPPN7nanojit4LInsEb(%"class.js::TraceRecorder"*, %struct.JSSpecializedNative*, i32, %"class.nanojit::LIns"**, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder21callSpecializedNativeEP17JSNativeTraceInfojb(%"class.js::TraceRecorder"*, %struct.JSNativeTraceInfo* nocapture, i32, i1 zeroext) nounwind align 2

declare i32 @strlen(i8* nocapture) nounwind readonly

declare x86_fastcallcc i32 @_ZN2js16ceilReturningIntEdPi(double, i32* nocapture) nounwind

declare x86_fastcallcc i32 @_ZN2js17floorReturningIntEdPi(double, i32* nocapture) nounwind

declare x86_fastcallcc i32 @_ZN2js17roundReturningIntEdPi(double, i32* nocapture) nounwind

declare hidden i32 @_ZN2js13TraceRecorder21callFloatReturningIntEjPKN7nanojit8CallInfoE(%"class.js::TraceRecorder"*, i32, %"struct.nanojit::CallInfo"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder10callNativeEj4JSOp(%"class.js::TraceRecorder"*, i32, i32) nounwind align 2

declare i32 @_Z12js_math_ceilP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z13js_math_floorP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z13js_math_roundP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare zeroext i1 @_ZN2js4tjit23IsPromotedInt32OrUint32EPN7nanojit4LInsE(%"class.nanojit::LIns"*)

declare i32 @_Z11js_math_absP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z13js_str_charAtP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare hidden i32 @_ZN2js13TraceRecorder9getCharAtEP8JSStringPN7nanojit4LInsES5_4JSOpPS5_(%"class.js::TraceRecorder"*, %struct.JSString* nocapture, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, i32, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare i32 @_Z17js_str_charCodeAtP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare hidden i32 @_ZN2js13TraceRecorder13getCharCodeAtEP8JSStringPN7nanojit4LInsES5_PS5_(%"class.js::TraceRecorder"*, %struct.JSString* nocapture, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare i32 @_Z14js_regexp_execP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare %struct.JSObject* @_ZN2js15HasNativeMethodEP8JSObjectiPFiP9JSContextjPNS_5ValueEE(%struct.JSObject*, i32, i32 (%struct.JSContext*, i32, %"class.js::Value"*)*)

declare i32 @_Z14js_regexp_testP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z11js_math_minP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z11js_math_maxP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z12js_fun_applyP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z11js_fun_callP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare hidden i32 @_ZN2js13TraceRecorder12functionCallEj4JSOp(%"class.js::TraceRecorder"*, i32, i32) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder11guardCalleeERNS_5ValueE(%"class.js::TraceRecorder"*, %"class.js::Value"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder23interpretedFunctionCallERNS_5ValueEP10JSFunctionjb(%"class.js::TraceRecorder"*, %"class.js::Value"*, %struct.JSFunction* nocapture, i32, i1 zeroext) nounwind align 2

declare i32 @_Z8js_ArrayP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_Z9js_StringP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare x86_fastcallcc i32 @_ZN2js12DeleteIntKeyEP9JSContextP8JSObjectii(%struct.JSContext*, %struct.JSObject*, i32, i32) nounwind

declare x86_fastcallcc i32 @_ZN2js12DeleteStrKeyEP9JSContextP8JSObjectP8JSStringi(%struct.JSContext*, %struct.JSObject*, %struct.JSString*, i32) nounwind

declare hidden i32 @_ZN2js13TraceRecorder7incNameEib(%"class.js::TraceRecorder"*, i32, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder4nameERPNS_5ValueERPN7nanojit4LInsERNS0_10NameResultE(%"class.js::TraceRecorder"*, %"class.js::Value"** nocapture, %"class.nanojit::LIns"** nocapture, %"struct.js::TraceRecorder::NameResult"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder11setCallPropEP8JSObjectPN7nanojit4LInsEPKNS_5ShapeES5_RKNS_5ValueE(%"class.js::TraceRecorder"*, %struct.JSObject* nocapture, %"class.nanojit::LIns"*, %"struct.js::Shape"* nocapture, %"class.nanojit::LIns"*, %"class.js::Value"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7getPropERNS_5ValueE(%"class.js::TraceRecorder"*, %"class.js::Value"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder22lookupForSetPropertyOpEP8JSObjectPN7nanojit4LInsEiPbPS2_PPKNS_5ShapeE(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, i32, i8* nocapture, %struct.JSObject** nocapture, %"struct.js::Shape"** nocapture) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js18MethodWriteBarrierEP9JSContextP8JSObjectjPKNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"* nocapture) nounwind

declare hidden i32 @_ZN2js13TraceRecorder9nativeSetEP8JSObjectPN7nanojit4LInsEPKNS_5ShapeERKNS_5ValueES5_(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, %"struct.js::Shape"* nocapture, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder15addDataPropertyEP8JSObject(%"class.js::TraceRecorder"* nocapture, %struct.JSObject*) nounwind readonly align 2

declare i32 @JS_PropertyStub(%struct.JSContext*, %struct.JSObject*, i32, i64*)

declare i32 @JS_StrictPropertyStub(%struct.JSContext*, %struct.JSObject*, i32, i32, i64*)

declare hidden i32 @_ZN2js13TraceRecorder18record_AddPropertyEP8JSObject(%"class.js::TraceRecorder"*, %struct.JSObject*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19setUpwardTrackedVarEPNS_5ValueERKS1_PN7nanojit4LInsE(%"class.js::TraceRecorder"*, %"class.js::Value"*, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden zeroext i8 @_ZN2js13TraceRecorder17determineSlotTypeEPNS_5ValueE(%"class.js::TraceRecorder"* nocapture, %"class.js::Value"*) nounwind inlinehint align 2

declare i32 @_ZN2js10SetCallArgEP9JSContextP8JSObjectiiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)

declare i32 @_ZN2js10SetCallVarEP9JSContextP8JSObjectiiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)

declare hidden i32 @_ZN2js13TraceRecorder11setPropertyEP8JSObjectPN7nanojit4LInsERKNS_5ValueES5_Pb(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, %"class.js::Value"* nocapture, %"class.nanojit::LIns"*, i8* nocapture) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder19recordSetPropertyOpEv(%"class.js::TraceRecorder"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder20recordInitPropertyOpEh(%"class.js::TraceRecorder"*, i8 zeroext) nounwind align 2

declare i32 @_Z22js_CheckForStringIndexi(i32)

declare hidden void @_ZN2js13TraceRecorder13finishGetPropEPN7nanojit4LInsES3_S3_PNS_5ValueE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.js::Value"*) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js17GetPropertyByNameEP9JSContextP8JSObjectPP8JSStringPNS_5ValueEPNS_8PICTableE(%struct.JSContext*, %struct.JSObject*, %struct.JSString** nocapture, %"class.js::Value"*, %"struct.js::PICTable"* nocapture) nounwind

declare hidden i32 @_ZN2js13TraceRecorder24primitiveToStringInPlaceEPNS_5ValueE(%"class.js::TraceRecorder"*, %"class.js::Value"*) nounwind align 2

declare %struct.JSString* @_Z16js_ValueToStringP9JSContextRKN2js5ValueE(%struct.JSContext*, %"class.js::Value"*)

declare hidden i32 @_ZN2js13TraceRecorder17getPropertyByNameEPN7nanojit4LInsEPNS_5ValueES5_(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.js::Value"*, %"class.js::Value"*) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js18GetPropertyByIndexEP9JSContextP8JSObjectiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*) nounwind

declare hidden i32 @_ZN2js13TraceRecorder18getPropertyByIndexEPN7nanojit4LInsES3_PNS_5ValueE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.js::Value"*) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js15GetPropertyByIdEP9JSContextP8JSObjectiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*) nounwind

declare hidden i32 @_ZN2js13TraceRecorder15getPropertyByIdEPN7nanojit4LInsEPNS_5ValueE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.js::Value"*) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js27GetPropertyWithNativeGetterEP9JSContextP8JSObjectPNS_5ShapeEPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, %"struct.js::Shape"* nocapture, %"class.js::Value"*) nounwind

declare hidden i32 @_ZN2js13TraceRecorder27getPropertyWithNativeGetterEPN7nanojit4LInsEPKNS_5ShapeEPNS_5ValueE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"struct.js::Shape"*, %"class.js::Value"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder27getPropertyWithScriptGetterEP8JSObjectPN7nanojit4LInsEPKNS_5ShapeE(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, %"struct.js::Shape"* nocapture) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder13getUnitStringEPN7nanojit4LInsES3_(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder12guardNotHoleEPN7nanojit4LInsES3_(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*) nounwind align 2

declare hidden %struct.JSStackFrame* @_ZN2js13TraceRecorder14guardArgumentsEP8JSObjectPN7nanojit4LInsEPj(%"class.js::TraceRecorder"*, %struct.JSObject* nocapture, %"class.nanojit::LIns"*, i32*) nounwind align 2

declare i32 @_Z15js_IsTypedArrayP8JSObject(%struct.JSObject*)

declare hidden i32 @_ZN2js13TraceRecorder17typedArrayElementERNS_5ValueES2_RPS1_RPN7nanojit4LInsE(%"class.js::TraceRecorder"*, %"class.js::Value"*, %"class.js::Value"*, %"class.js::Value"** nocapture, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js17SetPropertyByNameEP9JSContextP8JSObjectPP8JSStringPNS_5ValueEi(%struct.JSContext*, %struct.JSObject*, %struct.JSString** nocapture, %"class.js::Value"*, i32) nounwind

declare x86_fastcallcc i32 @_ZN2js18InitPropertyByNameEP9JSContextP8JSObjectPP8JSStringPKNS_5ValueE(%struct.JSContext*, %struct.JSObject*, %struct.JSString** nocapture, %"class.js::Value"*) nounwind

declare hidden i32 @_ZN2js13TraceRecorder23initOrSetPropertyByNameEPN7nanojit4LInsEPNS_5ValueES5_b(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.js::Value"*, %"class.js::Value"*, i1 zeroext) nounwind align 2

declare x86_fastcallcc i32 @_ZN2js18SetPropertyByIndexEP9JSContextP8JSObjectiPNS_5ValueEi(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32) nounwind

declare x86_fastcallcc i32 @_ZN2js19InitPropertyByIndexEP9JSContextP8JSObjectiPKNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*) nounwind

declare hidden i32 @_ZN2js13TraceRecorder24initOrSetPropertyByIndexEPN7nanojit4LInsES3_PNS_5ValueEb(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.js::Value"*, i1 zeroext) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7setElemEiii(%"class.js::TraceRecorder"*, i32, i32, i32) nounwind align 2

declare %"struct.js::TypedArray"* @_ZN2js10TypedArray12fromJSObjectEP8JSObject(%struct.JSObject*)

declare x86_fastcallcc i32 @_Z27js_EnsureDenseArrayCapacityP9JSContextP8JSObjecti(%struct.JSContext*, %struct.JSObject*, i32)

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder5upvarEP8JSScriptP12JSUpvarArrayjRNS_5ValueE(%"class.js::TraceRecorder"*, %struct.JSScript* nocapture, %struct.JSUpvarArray* nocapture, i32, %"class.js::Value"* nocapture) nounwind align 2

declare %"class.js::Value"* @_ZN2js8GetUpvarEP9JSContextjNS_11UpvarCookieE(%struct.JSContext*, i32, %"class.js::KidsPointer"* byval)

declare hidden i32 @_ZN2js13TraceRecorder10createThisER8JSObjectPN7nanojit4LInsEPS5_(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare %"struct.js::Shape"* @_ZN2js34LookupInterpretedFunctionPrototypeEP9JSContextP8JSObject(%struct.JSContext*, %struct.JSObject*)

declare hidden %"struct.js::FrameInfo"* @_ZN2js14FrameInfoCache7memoizeEPNS_9FrameInfoE(%"class.js::FrameInfoCache"* nocapture, %"struct.js::FrameInfo"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder26guardArgsLengthNotAssignedEPN7nanojit4LInsE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder25record_NativeCallCompleteEv(%"class.js::TraceRecorder"*) nounwind align 2

declare x86_fastcallcc %struct.JSObject* @_ZN2js17MethodReadBarrierEP9JSContextP8JSObjectPNS_5ShapeES3_(%struct.JSContext*, %struct.JSObject*, %"struct.js::Shape"*, %struct.JSObject*) nounwind

declare hidden i32 @_ZN2js13TraceRecorder8propTailEP8JSObjectPN7nanojit4LInsES2_NS_5PCValEPjPS5_PNS_5ValueE(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*, %struct.JSObject*, %"class.js::KidsPointer"* nocapture byval, i32*, %"class.nanojit::LIns"** nocapture, %"class.js::Value"*) nounwind align 2

declare hidden %"class.nanojit::LIns"* @_ZN2js13TraceRecorder16canonicalizeNaNsEPN7nanojit4LInsE(%"class.js::TraceRecorder"* nocapture, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder7getPropEP8JSObjectPN7nanojit4LInsE(%"class.js::TraceRecorder"*, %struct.JSObject*, %"class.nanojit::LIns"*) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder12getFullIndexEi(%"class.js::TraceRecorder"* nocapture, i32) nounwind readonly align 2

declare x86_fastcallcc i32 @_ZN2js16ObjectToIteratorEP9JSContextP8JSObjectiPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*) nounwind

declare x86_fastcallcc i32 @_ZN2js12IteratorMoreEP9JSContextP8JSObjectPNS_5ValueE(%struct.JSContext*, %struct.JSObject*, %"class.js::Value"*) nounwind

declare x86_fastcallcc i32 @_ZN2js13CloseIteratorEP9JSContextP8JSObject(%struct.JSContext*, %struct.JSObject*) nounwind

declare hidden void @_ZN2js13TraceRecorder10storeMagicE10JSWhyMagicNS_4tjit7AddressE(%"class.js::TraceRecorder"* nocapture, i32, %"struct.js::tjit::Address"* nocapture byval) nounwind align 2

declare hidden i32 @_ZN2js13TraceRecorder14unboxNextValueERPN7nanojit4LInsE(%"class.js::TraceRecorder"*, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare %struct.JSObject* @_Z21js_FindIdentifierBaseP9JSContextP8JSObjecti(%struct.JSContext*, %struct.JSObject*, i32)

declare x86_fastcallcc i32 @_ZN2js18HasInstanceOnTraceEP9JSContextP8JSObjectPKNS_5ValueE(%struct.JSContext*, %struct.JSObject*, %"class.js::Value"*) nounwind

declare i32 @_ZN2js10array_sortEP9JSContextjPNS_5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare i32 @_ZN2js11str_replaceEP9JSContextjPNS_5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare %struct.JSObject* @_ZN2js17GetBlockChainFastEP9JSContextP12JSStackFrame4JSOpm(%struct.JSContext*, %struct.JSStackFrame*, i32, i32)

declare hidden i32 @_ZN2js13TraceRecorder25record_DefLocalFunSetSlotEjP8JSObject(%"class.js::TraceRecorder"*, i32, %struct.JSObject* nocapture) nounwind align 2

declare hidden x86_fastcallcc i32 @_ZN2js10js_UnbrandEP9JSContextP8JSObject(%struct.JSContext*, %struct.JSObject*) nounwind

declare hidden void @_ZN2js13TraceRecorder17captureStackTypesEjP11JSValueType(%"class.js::TraceRecorder"* nocapture, i32, i8*) nounwind align 2

declare hidden void @_ZN2js13TraceRecorder20determineGlobalTypesEP11JSValueType(%"class.js::TraceRecorder"*, i8*) nounwind align 2

declare hidden i32 @_ZN2js16RecordTracePointEP9JSContextPNS_12TraceMonitorERjPbb(%struct.JSContext*, %"struct.js::TraceMonitor"*, i32*, i8*, i1 zeroext) nounwind

declare zeroext i1 @_ZN2js9InterpretEP9JSContextP12JSStackFramej12JSInterpMode(%struct.JSContext*, %struct.JSStackFrame*, i32, i32)

declare hidden void @_ZN2js11LoopProfileC1EPNS_12TraceMonitorEP12JSStackFramePhS5_(%"class.js::LoopProfile"*, %"struct.js::TraceMonitor"*, %struct.JSStackFrame*, i8*, i8*) nounwind align 2

declare hidden void @_ZN2js11LoopProfileC2EPNS_12TraceMonitorEP12JSStackFramePhS5_(%"class.js::LoopProfile"*, %"struct.js::TraceMonitor"*, %struct.JSStackFrame*, i8*, i8*) nounwind align 2

declare hidden void @_ZN2js11LoopProfile5resetEv(%"class.js::LoopProfile"* nocapture) nounwind align 2

declare hidden i32 @_ZN2js11LoopProfile15profileLoopEdgeEP9JSContextRj(%"class.js::LoopProfile"*, %struct.JSContext*, i32* nocapture) nounwind align 2

declare hidden void @_ZN2js11LoopProfile6decideEP9JSContext(%"class.js::LoopProfile"*, %struct.JSContext*) nounwind align 2

declare hidden void @_ZN2js11LoopProfile13stopProfilingEP9JSContext(%"class.js::LoopProfile"* nocapture, %struct.JSContext* nocapture) nounwind align 2

declare hidden i32 @_ZN2js17MonitorTracePointEP9JSContextRjPbPPvPjS6_j(%struct.JSContext*, i32*, i8*, i8** nocapture, i32* nocapture, i32* nocapture, i32) nounwind

declare hidden i32 @_ZN2js11LoopProfile16profileOperationEP9JSContext4JSOp(%"class.js::LoopProfile"*, %struct.JSContext*, i32) nounwind align 2

declare zeroext i1 @_Z17js_IsMathFunctionPFiP9JSContextjPyE(i32 (%struct.JSContext*, i32, i64*)*)

declare i32 @_ZN2js4mjit18GetCallTargetCountEP8JSScriptPh(%struct.JSScript*, i8*)

declare i32 @_Z17js_ValueToBooleanRKN2js5ValueE(%"class.js::Value"*)

declare hidden zeroext i1 @_ZN2js11LoopProfile22isCompilationExpensiveEP9JSContextj(%"class.js::LoopProfile"* nocapture, %struct.JSContext*, i32) nounwind align 2

declare hidden zeroext i1 @_ZN2js11LoopProfile25isCompilationUnprofitableEP9JSContextj(%"class.js::LoopProfile"* nocapture, %struct.JSContext* nocapture, i32) nounwind align 2

declare hidden i32 @_ZN2js15MonitorLoopEdgeEP9JSContextRj12JSInterpMode(%struct.JSContext*, i32*, i32) nounwind

declare hidden i32 @_ZN2js10GetHotloopEP9JSContext(%struct.JSContext* nocapture) nounwind readonly

declare fastcc void @_ZN2js15VisitFrameSlotsINS_19CaptureTypesVisitorEEEbRT_P9JSContextjP12JSStackFrameS7_(%"class.js::CaptureTypesVisitor"*, %struct.JSContext*, i32, %struct.JSStackFrame*, %struct.JSStackFrame*) nounwind

declare hidden zeroext i1 @_ZN2js6detail9HashTableINS_7HashMapIPN7nanojit4LInsEP8JSObjectNS_13DefaultHasherIS5_EENS_18ContextAllocPolicyEE5EntryENSB_13MapHashPolicyESA_E15changeTableSizeEi(%33* nocapture, i32) nounwind align 2

declare hidden %"class.js::detail::HashTable<js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry, js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::MapHashPolicy, js::ContextAllocPolicy>::Entry"* @_ZN2js6detail9HashTableINS_7HashMapIPN7nanojit4LInsEP8JSObjectNS_13DefaultHasherIS5_EENS_18ContextAllocPolicyEE5EntryENSB_13MapHashPolicyESA_E11createTableERSA_j(%"class.js::ContextAllocPolicy"* nocapture, i32) nounwind align 2

declare void @_ZN2js14GCHelperThread21replenishAndFreeLaterEPv(%"class.js::GCHelperThread"*, i8*)

declare i8* @_ZN9JSRuntime13onOutOfMemoryEPvmP9JSContext(%struct.JSRuntime*, i8*, i32, %struct.JSContext*)

declare void @_ZN9JSRuntime15onTooMuchMallocEv(%struct.JSRuntime*)

declare void @_Z27js_ReportAllocationOverflowP9JSContext(%struct.JSContext*)

declare hidden zeroext i1 @_ZN2js6detail9HashTableINS_7HashMapIPN7nanojit4LInsEP8JSObjectNS_13DefaultHasherIS5_EENS_18ContextAllocPolicyEE5EntryENSB_13MapHashPolicyESA_E3addERNSE_6AddPtrE(%33* nocapture, %"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::AddPtr"* nocapture) nounwind align 2

declare hidden void @_ZN2js6detail9HashTableINS_7HashMapIPN7nanojit4LInsEP8JSObjectNS_13DefaultHasherIS5_EENS_18ContextAllocPolicyEE5EntryENSB_13MapHashPolicyESA_E3Ptr7nonNullEv(%"class.js::detail::HashTable<js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::Entry, js::HashMap<nanojit::LIns *, JSObject *, js::DefaultHasher<nanojit::LIns *>, js::ContextAllocPolicy>::MapHashPolicy, js::ContextAllocPolicy>::Ptr"* nocapture) nounwind readnone align 2

declare hidden i64 @_ZNK2js6detail9HashTableINS_7HashMapIPN7nanojit4LInsEP8JSObjectNS_13DefaultHasherIS5_EENS_18ContextAllocPolicyEE5EntryENSB_13MapHashPolicyESA_E12lookupForAddERKS5_(%33* nocapture, %"class.nanojit::LIns"** nocapture) nounwind align 2

declare hidden zeroext i1 @_ZN2js6detail9HashTableINS_7HashMapIPhmNS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS7_13MapHashPolicyES6_E15changeTableSizeEi(%40* nocapture, i32) nounwind align 2

declare hidden zeroext i1 @_ZN2js6detail9HashTableIKP8JSScriptNS_7HashSetIS3_NS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE6SetOpsES8_E15changeTableSizeEi(%44* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js6detail9HashTableIKP8JSScriptNS_7HashSetIS3_NS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE6SetOpsES8_E3Ptr7nonNullEv(%"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Ptr"* nocapture) nounwind readnone align 2

declare void @_ZN7nanojit9AllocatorD2Ev(%"class.nanojit::Allocator"*)

declare void @_ZN7nanojit9CodeAllocD1Ev(%"class.nanojit::CodeAlloc"*)

declare void @free(i8* nocapture) nounwind

declare void @_ZN7nanojit9CodeAllocC1Ev(%"class.nanojit::CodeAlloc"*)

declare void @_ZN7nanojit9AllocatorC2Ev(%"class.nanojit::Allocator"*)

declare fastcc void @_ZN2js15VisitFrameSlotsINS_21DetermineTypesVisitorEEEbRT_P9JSContextjP12JSStackFrameS7_(%"class.js::DetermineTypesVisitor"*, %struct.JSContext*, i32, %struct.JSStackFrame*, %struct.JSStackFrame*) nounwind

declare %80 @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone

declare noalias i8* @realloc(i8* nocapture, i32) nounwind

declare fastcc void @_ZN2js15VisitFrameSlotsINS_17ClearSlotsVisitorEEEbRT_P9JSContextjP12JSStackFrameS7_(%"class.js::ClearSlotsVisitor"*, %struct.JSContext*, i32, %struct.JSStackFrame*, %struct.JSStackFrame*) nounwind

define linkonce_odr hidden void @_ZN2js5QueueINS_7SlotMap8SlotInfoEE6ensureEj(%67* nocapture %this, i32 %size) nounwind align 2 {
  br i1 undef, label %14, label %1

; <label>:1                                       ; preds = %0
  br i1 undef, label %2, label %3

; <label>:2                                       ; preds = %1
  br label %3

; <label>:3                                       ; preds = %2, %1
  br i1 undef, label %13, label %4

; <label>:4                                       ; preds = %3
  %5 = tail call %80 @llvm.umul.with.overflow.i32(i32 undef, i32 16)
  %6 = extractvalue %80 %5, 1
  %7 = extractvalue %80 %5, 0
  %.op = add i32 %7, 7
  %.op.op = and i32 %.op, -8
  %8 = select i1 %6, i32 0, i32 %.op.op
  br i1 undef, label %10, label %9

; <label>:9                                       ; preds = %4
  br label %_ZnamRN7nanojit9AllocatorE.exit

; <label>:10                                      ; preds = %4
  %11 = tail call i8* @_ZN7nanojit9Allocator9allocSlowEmb(%"class.nanojit::Allocator"* undef, i32 %8, i1 zeroext false) nounwind
  br label %_ZnamRN7nanojit9AllocatorE.exit

_ZnamRN7nanojit9AllocatorE.exit:                  ; preds = %10, %9
  br i1 false, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %_ZnamRN7nanojit9AllocatorE.exit
  br label %12

; <label>:12                                      ; preds = %12, %.lr.ph
  br i1 undef, label %._crit_edge, label %12

._crit_edge:                                      ; preds = %12, %_ZnamRN7nanojit9AllocatorE.exit
  br label %14

; <label>:13                                      ; preds = %3
  br label %14

; <label>:14                                      ; preds = %13, %._crit_edge, %0
  ret void
}

declare hidden void @_ZN2js5QueueIPNS_12TreeFragmentEE3addES2_(%"class.js::Queue"* nocapture, %"struct.js::TreeFragment"*) nounwind align 2

declare hidden zeroext i1 @_ZN2js6VectorI11JSValueTypeLm256ENS_18ContextAllocPolicyEE10growByImplILb1EEEbm(%35*, i32) nounwind inlinehint align 2

declare hidden zeroext i1 @_ZN2js6VectorI11JSValueTypeLm256ENS_18ContextAllocPolicyEE20convertToHeapStorageEm(%35* nocapture, i32) nounwind align 2

declare hidden zeroext i1 @_ZN2js10VectorImplI11JSValueTypeLm256ENS_18ContextAllocPolicyELb0EE6growToERNS_6VectorIS1_Lm256ES2_EEm(%35* nocapture, i32) nounwind inlinehint align 2

declare i32 @llvm.ctlz.i32(i32) nounwind readnone

declare hidden zeroext i1 @_ZN2js6VectorIjLm0ENS_18ContextAllocPolicyEE20convertToHeapStorageEm(%34* nocapture, i32) nounwind align 2

declare hidden zeroext i1 @_ZN2js6VectorIjLm0ENS_18ContextAllocPolicyEE17growHeapStorageByEm(%34* nocapture, i32) nounwind align 2

declare fastcc void @_ZN2js15VisitFrameSlotsINS_27ImportBoxedStackSlotVisitorEEEbRT_P9JSContextjP12JSStackFrameS7_(%"class.js::ImportBoxedStackSlotVisitor"*, %struct.JSContext*, i32, %struct.JSStackFrame*, %struct.JSStackFrame*) nounwind

declare hidden zeroext i1 @_ZN2js6detail9HashTableINS_7HashMapIPhPNS_11LoopProfileENS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS9_13MapHashPolicyES8_E15changeTableSizeEi(%42* nocapture, i32) nounwind align 2

declare fastcc zeroext i1 @_ZN2js15VisitFrameSlotsINS_17CountSlotsVisitorEEEbRT_P9JSContextjP12JSStackFrameS7_(%"struct.js::CountSlotsVisitor"*, %struct.JSContext*, i32, %struct.JSStackFrame*, %struct.JSStackFrame*) nounwind

declare hidden void @_ZN2js5QueueINS_5ValueEE3addES1_(%18* nocapture, %"class.js::Value"* nocapture byval align 4) nounwind align 2

declare hidden void @_ZN2js6detail9HashTableINS_7HashMapIPhPNS_11LoopProfileENS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS9_13MapHashPolicyES8_E3Ptr7nonNullEv(%"class.js::detail::HashTable<js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, js::LoopProfile *, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Ptr"* nocapture) nounwind readnone align 2

declare hidden zeroext i1 @_ZN2js6detail9HashTableINS_7HashMapIPhPNS_11LoopProfileENS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS9_13MapHashPolicyES8_E3addERNSC_6AddPtrE(%42* nocapture, %"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::AddPtr"* nocapture) nounwind align 2

declare hidden i64 @_ZNK2js6detail9HashTableINS_7HashMapIPhPNS_11LoopProfileENS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS9_13MapHashPolicyES8_E12lookupForAddERKS3_(%42* nocapture, i8** nocapture) nounwind align 2

declare void @_ZN8JSObject16generateOwnShapeEP9JSContext(%struct.JSObject*, %struct.JSContext*)

declare i32 @_ZN2js11HasInstanceEP9JSContextP8JSObjectPKNS_5ValueEPi(%struct.JSContext*, %struct.JSObject*, %"class.js::Value"*, i32*)

declare i32 @_Z17js_LookupPropertyP9JSContextP8JSObjectiPS2_PP10JSProperty(%struct.JSContext*, %struct.JSObject*, i32, %struct.JSObject**, %struct.JSProperty**)

declare %struct.JSAtom* @_Z16js_AtomizeStringP9JSContextP8JSStringj(%struct.JSContext*, %struct.JSString*, i32)

declare x86_fastcallcc %struct.JSString* @_Z17js_NumberToStringP9JSContextd(%struct.JSContext*, double)

declare i32 @_Z16js_CloseIteratorP9JSContextP8JSObject(%struct.JSContext*, %struct.JSObject*)

declare i32 @_Z15js_IteratorMoreP9JSContextP8JSObjectPN2js5ValueE(%struct.JSContext*, %struct.JSObject*, %"class.js::Value"*)

declare i32 @_Z18js_ValueToIteratorP9JSContextjPN2js5ValueE(%struct.JSContext*, i32, %"class.js::Value"*)

declare %"class.nanojit::LIns"* @_ZN7nanojit9LirWriter9insChooseEPNS_4LInsES2_S2_b(%"class.nanojit::LirFilter"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, i1 zeroext)

declare %"struct.js::Shape"* @_ZN8JSObject17methodShapeChangeEP9JSContextRKN2js5ShapeE(%struct.JSObject*, %struct.JSContext*, %"struct.js::Shape"*)

declare x86_fastcallcc %struct.JSObject* @_Z22js_CloneFunctionObjectP9JSContextP10JSFunctionP8JSObjectS4_(%struct.JSContext*, %struct.JSFunction*, %struct.JSObject*, %struct.JSObject*)

declare %"class.nanojit::LIns"* @_ZN7nanojit9LirWriter8insStoreEPNS_4LInsES2_ij(%"class.nanojit::LirFilter"*, %"class.nanojit::LIns"*, %"class.nanojit::LIns"*, i32, i32)

declare hidden zeroext i1 @_ZN2js6detail9HashTableIKPNS_9FrameInfoENS_7HashSetIS3_NS_14FrameInfoCache10HashPolicyENS_17SystemAllocPolicyEE6SetOpsES8_E3addERNSB_6AddPtrE(%30* nocapture, %"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::AddPtr"* nocapture) nounwind align 2

declare hidden zeroext i1 @_ZN2js6detail9HashTableIKPNS_9FrameInfoENS_7HashSetIS3_NS_14FrameInfoCache10HashPolicyENS_17SystemAllocPolicyEE6SetOpsES8_E15changeTableSizeEi(%30* nocapture, i32) nounwind align 2

declare hidden void @_ZN2js6detail9HashTableIKPNS_9FrameInfoENS_7HashSetIS3_NS_14FrameInfoCache10HashPolicyENS_17SystemAllocPolicyEE6SetOpsES8_E3Ptr7nonNullEv(%"class.js::detail::HashTable<js::FrameInfo *const, js::HashSet<js::FrameInfo *, js::FrameInfoCache::HashPolicy, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Ptr"* nocapture) nounwind readnone align 2

declare hidden %"class.js::detail::HashTable<js::FrameInfo *const, js::HashSet<js::FrameInfo *, js::FrameInfoCache::HashPolicy, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::Entry"* @_ZNK2js6detail9HashTableIKPNS_9FrameInfoENS_7HashSetIS3_NS_14FrameInfoCache10HashPolicyENS_17SystemAllocPolicyEE6SetOpsES8_E6lookupERS4_jj(%30* nocapture, %"struct.js::FrameInfo"** nocapture, i32, i32) nounwind align 2

declare i32 @_Z17js_DefinePropertyP9JSContextP8JSObjectiPKN2js5ValueEPFiS0_S2_iPS4_EPFiS0_S2_iiS7_Ej(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32 (%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*)*, i32 (%struct.JSContext*, %struct.JSObject*, i32, i32, %"class.js::Value"*)*, i32)

declare void @_ZN2js10LeaveTraceEP9JSContext(%struct.JSContext*)

declare i32 @_Z14js_SetPropertyP9JSContextP8JSObjectiPN2js5ValueEi(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32)

declare zeroext i1 @_Z29js_GetPropertyHelperWithShapeP9JSContextP8JSObjectS2_ijPN2js5ValueEPPKNS3_5ShapeEPS2_(%struct.JSContext*, %struct.JSObject*, %struct.JSObject*, i32, i32, %"class.js::Value"*, %"struct.js::Shape"**, %struct.JSObject**)

declare %"struct.js::Shape"** @_ZN2js13PropertyTable6searchEib(%"struct.js::PropertyTable"*, i32, i1 zeroext)

declare zeroext i1 @_ZN2js5Shape7hashifyEP9JSRuntime(%"struct.js::Shape"*, %struct.JSRuntime*)

declare hidden void @_ZN2js5QueueIPKNS_5ShapeEE3addES3_(%19* nocapture, %"struct.js::Shape"*) nounwind align 2

declare zeroext i1 @_ZN8JSObject17methodShapeChangeEP9JSContextj(%struct.JSObject*, %struct.JSContext*, i32)

declare i32 @JS_ResolveStub(%struct.JSContext*, %struct.JSObject*, i32)

declare i32 @_Z17js_DeletePropertyP9JSContextP8JSObjectiPN2js5ValueEi(%struct.JSContext*, %struct.JSObject*, i32, %"class.js::Value"*, i32)

declare double @_Z18js_math_round_impld(double)

declare double @_Z18js_math_floor_impld(double)

declare double @_Z17js_math_ceil_impld(double)

declare %"class.js::MathCache"* @_ZN13JSCompartment14allocMathCacheEP9JSContext(%struct.JSCompartment*, %struct.JSContext*)

declare i32 @_Z14js_GetPropertyP9JSContextP8JSObjectS2_iPN2js5ValueE(%struct.JSContext*, %struct.JSObject*, %struct.JSObject*, i32, %"class.js::Value"*)

declare zeroext i1 @_ZN2js12BoxThisForVpEP9JSContextPNS_5ValueE(%struct.JSContext*, %"class.js::Value"*)

declare i32 @JS_GetTrapOpcode(%struct.JSContext*, %struct.JSScript*, i8*)

declare %struct.JSAtom* @_ZN2js13PropertyCache8fullTestEP9JSContextPhPP8JSObjectS6_PNS_18PropertyCacheEntryE(%"class.js::PropertyCache"*, %struct.JSContext*, i8*, %struct.JSObject**, %struct.JSObject**, %"struct.js::PropertyCacheEntry"*)

declare zeroext i1 @_ZN2js14CompareStringsEP9JSContextP8JSStringS3_Pi(%struct.JSContext*, %struct.JSString*, %struct.JSString*, i32*)

declare zeroext i1 @_ZN2js17ValueToNumberSlowEP9JSContextNS_5ValueEPd(%struct.JSContext*, %"class.js::Value"* byval align 4, double*)

declare %"class.nanojit::LIns"* @_ZN2js4tjit14DemoteToUint32EPN7nanojit9LirWriterEPNS1_4LInsE(%"class.nanojit::LirFilter"*, %"class.nanojit::LIns"*)

declare i32 @_ZNK7nanojit8CallInfo10count_argsEv(%"struct.nanojit::CallInfo"*)

declare void @_ZN7nanojit8Interval2ofEPNS_4LInsEi(%"struct.nanojit::Interval"* sret, %"class.nanojit::LIns"*, i32)

declare void @_ZN7nanojit8Interval3addES0_S0_(%"struct.nanojit::Interval"* sret, %"struct.nanojit::Interval"* byval, %"struct.nanojit::Interval"* byval)

declare void @_ZN7nanojit8Interval3subES0_S0_(%"struct.nanojit::Interval"* sret, %"struct.nanojit::Interval"* byval, %"struct.nanojit::Interval"* byval)

declare void @_ZN7nanojit8Interval3mulES0_S0_(%"struct.nanojit::Interval"* sret, %"struct.nanojit::Interval"* byval, %"struct.nanojit::Interval"* byval)

declare fastcc void @_ZN2js15SynthesizeFrameEP9JSContextRKNS_9FrameInfoEP8JSObject(%struct.JSContext*, %"struct.js::FrameInfo"* nocapture, %struct.JSObject* nocapture) nounwind

declare void @_Z25js_ReportOutOfScriptQuotaP9JSContext(%struct.JSContext*)

declare fastcc void @_ZN2js15VisitFrameSlotsINS_28FlushNativeStackFrameVisitorEEEbRT_P9JSContextjP12JSStackFrameS7_(%"class.js::FlushNativeStackFrameVisitor"*, %struct.JSContext*, i32, %struct.JSStackFrame*, %struct.JSStackFrame*) nounwind

declare i8* @getenv(i8* nocapture) nounwind readonly

declare i32 @strcmp(i8* nocapture, i8* nocapture) nounwind readonly

declare hidden void @_ZN2js5QueueI11JSValueTypeE3addEPS1_j(%78* nocapture, i8* nocapture, i32) nounwind align 2

declare void @_ZN7nanojit8FragmentC2EPKv(%"class.nanojit::Fragment"*, i8*)

declare void @_Z20js_ReportOutOfMemoryP9JSContext(%struct.JSContext*)

declare hidden zeroext i1 @_ZN2js6detail9HashTableIKP8JSScriptNS_7HashSetIS3_NS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE6SetOpsES8_E3addERNSB_6AddPtrE(%44* nocapture, %"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::AddPtr"* nocapture) nounwind align 2

declare hidden i64 @_ZNK2js6detail9HashTableIKP8JSScriptNS_7HashSetIS3_NS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE6SetOpsES8_E12lookupForAddERS4_(%44* nocapture, %struct.JSScript** nocapture) nounwind align 2

declare zeroext i1 @_ZN8JSObject26globalObjectOwnShapeChangeEP9JSContext(%struct.JSObject*, %struct.JSContext*)

declare noalias i8* @malloc(i32) nounwind

declare hidden void @_ZN2js14DefaultSlotMapD0Ev(%"class.js::DefaultSlotMap"*) nounwind align 2

declare hidden void @_ZN2js7SlotMap11adjustTypesEv(%"class.js::SlotMap"*) nounwind align 2

declare hidden void @_ZN2js7SlotMap10adjustTypeERNS0_8SlotInfoE(%"class.js::SlotMap"* nocapture, %"struct.js::SlotMap::SlotInfo"* nocapture) nounwind align 2

declare void @_ZdlPv(i8*) nounwind

declare hidden void @_ZN2js7SlotMapD1Ev(%"class.js::SlotMap"* nocapture) nounwind align 2

declare hidden void @_ZN2js7SlotMapD0Ev(%"class.js::SlotMap"*) nounwind align 2

declare hidden void @_ZN2js6detail9HashTableINS_7HashMapIPhmNS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS7_13MapHashPolicyES6_E3Ptr7nonNullEv(%"class.js::detail::HashTable<js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::Entry, js::HashMap<unsigned char *, unsigned long, js::DefaultHasher<unsigned char *>, js::SystemAllocPolicy>::MapHashPolicy, js::SystemAllocPolicy>::Ptr"* nocapture) nounwind readnone align 2

declare i8* @_ZN7nanojit9Allocator9allocSlowEmb(%"class.nanojit::Allocator"*, i32, i1 zeroext)

declare %"class.nanojit::LIns"* @_ZN2js4tjit13DemoteToInt32EPN7nanojit9LirWriterEPNS1_4LInsE(%"class.nanojit::LirFilter"*, %"class.nanojit::LIns"*)

declare void @_ZN2js8GCMarker20delayMarkingChildrenEPv(%"struct.js::GCMarker"*, i8*)

declare void @js_TraceXML(%struct.JSTracer*, %struct.JSXML*)

declare fastcc void @_ZN2js2gc10MarkObjectEP8JSTracerR8JSObjectPKc(%struct.JSTracer*, %struct.JSObject*) nounwind inlinehint

declare void @_ZNK2js5Shape5traceEP8JSTracer(%"struct.js::Shape"*, %struct.JSTracer*)

declare void @_Z14js_TraceObjectP8JSTracerP8JSObject(%struct.JSTracer*, %struct.JSObject*)

declare zeroext i1 @_Z20IsAboutToBeFinalizedP9JSContextPv(%struct.JSContext*, i8*)

declare hidden zeroext i1 @_ZN2js6detail9HashTableINS_7HashMapIPhmNS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS7_13MapHashPolicyES6_E3addERNSA_6AddPtrE(%40* nocapture, %"class.js::detail::HashTable<JSScript *const, js::HashSet<JSScript *, js::DefaultHasher<JSScript *>, js::SystemAllocPolicy>::SetOps, js::SystemAllocPolicy>::AddPtr"* nocapture) nounwind align 2

declare hidden i64 @_ZNK2js6detail9HashTableINS_7HashMapIPhmNS_13DefaultHasherIS3_EENS_17SystemAllocPolicyEE5EntryENS7_13MapHashPolicyES6_E12lookupForAddERKS3_(%40* nocapture, i8** nocapture) nounwind align 2

declare noalias i8* @calloc(i32, i32) nounwind

declare hidden void @_ZN7nanojit10LogControlD0Ev(%"class.nanojit::LogControl"*) nounwind align 2

declare void @_GLOBAL__I_a() nounwind section "__TEXT,__StaticInit,regular,pure_instructions"

declare %80 @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone

declare void @memset_pattern16(i8*, i8*, i32)
