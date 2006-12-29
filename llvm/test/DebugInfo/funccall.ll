;; RUN: llvm-upgrade < %s | llvm-as | llc

;; Debugger type declarations
%llvm.dbg.anchor.type = type { uint, uint }
%llvm.dbg.basictype.type = type { uint, {  }*, sbyte*, {  }*, uint, ulong, ulong, ulong, uint, uint }
%llvm.dbg.compile_unit.type = type { uint, {  }*, uint, sbyte*, sbyte*, sbyte* }
%llvm.dbg.global_variable.type = type { uint, {  }*, {  }*, sbyte*, sbyte*, sbyte*, {  }*, uint, {  }*, bool, bool, {  }* }
%llvm.dbg.subprogram.type = type { uint, {  }*, {  }*, sbyte*, sbyte*, sbyte*, {  }*, uint, {  }*, bool, bool }
%llvm.dbg.variable.type = type { uint, {  }*, sbyte*, {  }*, uint, {  }* }

;; Debugger intrinsic declarations...
declare void %llvm.dbg.func.start({  }*)
declare void %llvm.dbg.stoppoint(uint, uint, {  }*)
declare void %llvm.dbg.declare({  }*, {  }*)
declare void %llvm.dbg.region.start({  }*)
declare void %llvm.dbg.region.end({  }*)

;; Debugger anchors
%llvm.dbg.subprograms = linkonce constant %llvm.dbg.anchor.type {
  uint 393216,                                                                   ;; DW_TAG_anchor | version(6)
  uint 46 }, section "llvm.metadata"                                             ;; DW_TAG_subprogram
%llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type {
  uint 393216,                                                                   ;; DW_TAG_anchor | version(6)
  uint 17 }, section "llvm.metadata"                                             ;; DW_TAG_compile_unit
%llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type {
  uint 393216,                                                                   ;; DW_TAG_anchor | version(6)
  uint 52 }, section "llvm.metadata"                                             ;; DW_TAG_variable

;; Debug info
%llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type {
    uint 393262,                                                                 ;; DW_TAG_subprogram | version(6)
    {  }* bitcast (%llvm.dbg.anchor.type* %llvm.dbg.subprograms to {  }*),       ;; Anchor
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Context
    sbyte* getelementptr ([4 x sbyte]* %str, int 0, int 0),                      ;; Name
    sbyte* getelementptr ([4 x sbyte]* %str, int 0, int 0),                      ;; Fully quanlified name
    sbyte* null,                                                                 ;; Linkage name
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Compile unit
    uint 4,                                                                      ;; Line number
    {  }* null,                                                                  ;; Type descriptor
    bool false,                                                                  ;; Static?
    bool true }, section "llvm.metadata"                                         ;; External?
%str = internal constant [4 x sbyte] c"foo\00", section "llvm.metadata"
    
%llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type {
    uint 393233,                                                                 ;; DW_TAG_compile_unit | version(6)
    {  }* bitcast (%llvm.dbg.anchor.type* %llvm.dbg.compile_units to {  }*),     ;; Anchor
    uint 1,                                                                      ;; Language
    sbyte* getelementptr ([11 x sbyte]* %str, int 0, int 0),                     ;; Source file
    sbyte* getelementptr ([50 x sbyte]* %str, int 0, int 0),                     ;; Source file directory
    sbyte* getelementptr ([45 x sbyte]* %str, int 0, int 0) }, section "llvm.metadata" ;; Produceer
%str = internal constant [11 x sbyte] c"funccall.c\00", section "llvm.metadata"
%str = internal constant [50 x sbyte] c"/Volumes/Big2/llvm/llvm/test/Regression/Debugger/\00", section "llvm.metadata"
%str = internal constant [45 x sbyte] c"4.0.1 LLVM (Apple Computer, Inc. build 5421)\00", section "llvm.metadata"

%llvm.dbg.variable = internal constant %llvm.dbg.variable.type {
    uint 393472,                                                                 ;; DW_TAG_auto_variable | version(6)
    {  }* bitcast (%llvm.dbg.subprogram.type* %llvm.dbg.subprogram to {  }*),    ;; Context
    sbyte* getelementptr ([2 x sbyte]* %str, int 0, int 0),                      ;; Name
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Compile unit
    uint 5,                                                                      ;; Line number
    {  }* bitcast (%llvm.dbg.basictype.type* %llvm.dbg.basictype to {  }*) }, section "llvm.metadata" ;; Type
%str = internal constant [2 x sbyte] c"t\00", section "llvm.metadata"

%llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type {
    uint 393252,                                                                 ;; DW_TAG_base_type | version(6)
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Context
    sbyte* getelementptr ([4 x sbyte]* %str1, int 0, int 0),                     ;; Name
    {  }* null,                                                                  ;; Compile Unit
    uint 0,                                                                      ;; Line number
    ulong 32,                                                                    ;; Size in bits
    ulong 32,                                                                    ;; Align in bits
    ulong 0,                                                                     ;; Offset in bits
    uint 0,                                                                      ;; Flags
    uint 5 }, section "llvm.metadata"                                            ;; Basic type encoding
%str1 = internal constant [4 x sbyte] c"int\00", section "llvm.metadata"

%llvm.dbg.subprogram2 = internal constant %llvm.dbg.subprogram.type {
    uint 393262,                                                                 ;; DW_TAG_subprogram | version(6)
    {  }* bitcast (%llvm.dbg.anchor.type* %llvm.dbg.subprograms to {  }*),       ;; Anchor
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Context
    sbyte* getelementptr ([5 x sbyte]* %str, int 0, int 0),                      ;; Name
    sbyte* getelementptr ([5 x sbyte]* %str, int 0, int 0),                      ;; Fully quanlified name
    sbyte* null,                                                                 ;; Linkage name
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Compile unit
    uint 8,                                                                      ;; Line number
    {  }* bitcast (%llvm.dbg.basictype.type* %llvm.dbg.basictype to {  }*),      ;; Type descriptor
    bool false,                                                                  ;; Static?
    bool true }, section "llvm.metadata"                                         ;; External?
%str = internal constant [5 x sbyte] c"main\00", section "llvm.metadata"

%llvm.dbg.variable3 = internal constant %llvm.dbg.variable.type {
    uint 393474,                                                                 ;; DW_TAG_return_variable | version(6)
    {  }* bitcast (%llvm.dbg.subprogram.type* %llvm.dbg.subprogram2 to {  }*),   ;; Context
    sbyte* getelementptr ([7 x sbyte]* %str, int 0, int 0),                      ;; Name
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Compile unit
    uint 8,                                                                      ;; Line number
    {  }* bitcast (%llvm.dbg.basictype.type* %llvm.dbg.basictype to {  }*) }, section "llvm.metadata" ;; Type 
%str = internal constant [7 x sbyte] c"retval\00", section "llvm.metadata"

%llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type {
    uint 393268,                                                                 ;; DW_TAG_variable | version(6)
    {  }* bitcast (%llvm.dbg.anchor.type* %llvm.dbg.global_variables to {  }*),  ;; Anchor
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Context
    sbyte* getelementptr ([2 x sbyte]* %str4, int 0, int 0),                     ;; Name
    sbyte* getelementptr ([2 x sbyte]* %str4, int 0, int 0),                     ;; Fully qualified name
    sbyte* null,                                                                 ;; Linkage name
    {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*),;; Compile unit
    uint 2,                                                                      ;; Line number
    {  }* bitcast (%llvm.dbg.basictype.type* %llvm.dbg.basictype to {  }*),      ;; Type
    bool true,                                                                   ;; Static?
    bool true,                                                                   ;; External?
    {  }* bitcast (int* %q to {  }*) }, section "llvm.metadata"                  ;; Variable
%str4 = internal constant [2 x sbyte] c"q\00", section "llvm.metadata"

;; Global
%q = internal global int 0

implementation

void %foo() {
entry:
	%t = alloca int, align 4
	"alloca point" = bitcast int 0 to int
	call void %llvm.dbg.func.start( {  }* bitcast (%llvm.dbg.subprogram.type* %llvm.dbg.subprogram to {  }*) )
	call void %llvm.dbg.stoppoint( uint 4, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	%t = bitcast int* %t to {  }*
	call void %llvm.dbg.declare( {  }* %t, {  }* bitcast (%llvm.dbg.variable.type* %llvm.dbg.variable to {  }*) )
	call void %llvm.dbg.stoppoint( uint 5, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	%tmp = load int* %q
	store int %tmp, int* %t
	call void %llvm.dbg.stoppoint( uint 6, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	%tmp1 = load int* %t
	%tmp2 = add int %tmp1, 1
	store int %tmp2, int* %q
	call void %llvm.dbg.stoppoint( uint 7, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	call void %llvm.dbg.region.end( {  }* bitcast (%llvm.dbg.subprogram.type* %llvm.dbg.subprogram to {  }*) )
	ret void
}

int %main() {
entry:
	%retval = alloca int, align 4
	%tmp = alloca int, align 4
	"alloca point" = bitcast int 0 to int
	call void %llvm.dbg.func.start( {  }* bitcast (%llvm.dbg.subprogram.type* %llvm.dbg.subprogram2 to {  }*) )
	call void %llvm.dbg.stoppoint( uint 8, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	%retval = bitcast int* %retval to {  }*
	call void %llvm.dbg.declare( {  }* %retval, {  }* bitcast (%llvm.dbg.variable.type* %llvm.dbg.variable3 to {  }*) )
	call void %llvm.dbg.stoppoint( uint 9, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	store int 0, int* %q
	call void %llvm.dbg.stoppoint( uint 10, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	call void (...)* bitcast (void ()* %foo to void (...)*)( )
	call void %llvm.dbg.stoppoint( uint 11, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	%tmp = load int* %q
	%tmp1 = sub int %tmp, 1
	store int %tmp1, int* %q
	call void %llvm.dbg.stoppoint( uint 13, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	%tmp2 = load int* %q
	store int %tmp2, int* %tmp
	%tmp3 = load int* %tmp
	store int %tmp3, int* %retval
	%retval = load int* %retval
	call void %llvm.dbg.stoppoint( uint 14, uint 0, {  }* bitcast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*) )
	call void %llvm.dbg.region.end( {  }* bitcast (%llvm.dbg.subprogram.type* %llvm.dbg.subprogram2 to {  }*) )
	ret int %retval
}
