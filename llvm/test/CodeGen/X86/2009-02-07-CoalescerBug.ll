; RUN: llvm-as < %s | llc -march=x86 -relocation-model=pic -stats |& grep {Number of valno def marked dead} | grep 1
; rdar://6566708

target triple = "i386-apple-darwin9.6"
	%"struct..0$_58" = type { i32, %"struct.llvm::MachineOperand"**, %"struct.llvm::MachineOperand"* }
	%"struct..1$_60" = type { i32 }
	%"struct..3$_53" = type { i64 }
	%struct.__false_type = type <{ i8 }>
	%"struct.llvm::APFloat" = type { %"struct.llvm::fltSemantics"*, %"struct..3$_53", i16, i16 }
	%"struct.llvm::AbstractTypeUser" = type { i32 (...)** }
	%"struct.llvm::AnalysisResolver" = type { %"struct.std::vector<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >", %"struct.llvm::PMDataManager"* }
	%"struct.llvm::Annotable" = type { %"struct.llvm::Annotation"* }
	%"struct.llvm::Annotation" = type { i32 (...)**, %"struct..1$_60", %"struct.llvm::Annotation"* }
	%"struct.llvm::Argument" = type { %"struct.llvm::Value", %"struct.llvm::ilist_node<llvm::Argument>", %"struct.llvm::Function"* }
	%"struct.llvm::AttrListPtr" = type { %"struct.llvm::AttributeListImpl"* }
	%"struct.llvm::AttributeListImpl" = type opaque
	%"struct.llvm::BasicBlock" = type { %"struct.llvm::Value", %"struct.llvm::ilist_node<llvm::BasicBlock>", %"struct.llvm::iplist<llvm::Instruction,llvm::ilist_traits<llvm::Instruction> >", %"struct.llvm::Function"* }
	%"struct.llvm::BitVector" = type { i32*, i32, i32 }
	%"struct.llvm::BumpPtrAllocator" = type { i8* }
	%"struct.llvm::CalleeSavedInfo" = type { i32, %"struct.llvm::TargetRegisterClass"*, i32 }
	%"struct.llvm::CondCodeSDNode" = type { %"struct.llvm::SDNode", i32 }
	%"struct.llvm::Constant" = type { %"struct.llvm::User" }
	%"struct.llvm::DebugLocTracker" = type { %"struct.std::vector<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >", %"struct.llvm::DenseMap<llvm::DebugLocTuple,unsigned int,llvm::DenseMapInfo<llvm::DebugLocTuple>,llvm::DenseMapInfo<unsigned int> >" }
	%"struct.llvm::DebugLocTuple" = type { i32, i32, i32 }
	%"struct.llvm::DenseMap<llvm::DebugLocTuple,unsigned int,llvm::DenseMapInfo<llvm::DebugLocTuple>,llvm::DenseMapInfo<unsigned int> >" = type { i32, %"struct.std::pair<llvm::DebugLocTuple,unsigned int>"*, i32, i32 }
	%"struct.llvm::DwarfWriter" = type opaque
	%"struct.llvm::FoldingSet<llvm::SDNode>" = type { %"struct.llvm::FoldingSetImpl" }
	%"struct.llvm::FoldingSetImpl" = type { i32 (...)**, i8**, i32, i32 }
	%"struct.llvm::Function" = type { %"struct.llvm::GlobalValue", %"struct.llvm::Annotable", %"struct.llvm::ilist_node<llvm::Function>", %"struct.llvm::iplist<llvm::BasicBlock,llvm::ilist_traits<llvm::BasicBlock> >", %"struct.llvm::iplist<llvm::Argument,llvm::ilist_traits<llvm::Argument> >", %"struct.llvm::ValueSymbolTable"*, %"struct.llvm::AttrListPtr" }
	%"struct.llvm::FunctionLoweringInfo" = type opaque
	%"struct.llvm::GlobalAddressSDNode" = type { %"struct.llvm::SDNode", %"struct.llvm::GlobalValue"*, i64 }
	%"struct.llvm::GlobalValue" = type { %"struct.llvm::Constant", %"struct.llvm::Module"*, i32, %"struct.std::string" }
	%"struct.llvm::GlobalVariable" = type { %"struct.llvm::GlobalValue", %"struct.llvm::ilist_node<llvm::GlobalVariable>", i8 }
	%"struct.llvm::ImmutablePass" = type { %"struct.llvm::ModulePass" }
	%"struct.llvm::Instruction" = type { %"struct.llvm::User", %"struct.llvm::ilist_node<llvm::Instruction>", %"struct.llvm::BasicBlock"* }
	%"struct.llvm::LandingPadInfo" = type <{ %"struct.llvm::MachineBasicBlock"*, [12 x i8], %"struct.llvm::SmallVector<unsigned int,1u>", %"struct.llvm::SmallVector<unsigned int,1u>", i32, %"struct.llvm::Function"*, %"struct.std::vector<int,std::allocator<int> >", [3 x i32] }>
	%"struct.llvm::MVT" = type { %"struct..1$_60" }
	%"struct.llvm::MachineBasicBlock" = type { %"struct.llvm::ilist_node<llvm::MachineBasicBlock>", %"struct.llvm::ilist<llvm::MachineInstr>", %"struct.llvm::BasicBlock"*, i32, %"struct.llvm::MachineFunction"*, %"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >", %"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >", %"struct.std::vector<int,std::allocator<int> >", i32, i8 }
	%"struct.llvm::MachineConstantPool" = type opaque
	%"struct.llvm::MachineFrameInfo" = type { %"struct.std::vector<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >", i32, i8, i8, i64, i32, i32, i8, i32, i32, %"struct.std::vector<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >", %"struct.llvm::MachineModuleInfo"*, %"struct.llvm::TargetFrameInfo"* }
	%"struct.llvm::MachineFrameInfo::StackObject" = type { i64, i32, i8, i64 }
	%"struct.llvm::MachineFunction" = type { %"struct.llvm::Annotation", %"struct.llvm::Function"*, %"struct.llvm::TargetMachine"*, %"struct.llvm::MachineRegisterInfo"*, %"struct.llvm::AbstractTypeUser"*, %"struct.llvm::MachineFrameInfo"*, %"struct.llvm::MachineConstantPool"*, %"struct.llvm::MachineJumpTableInfo"*, %"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >", %"struct.llvm::BumpPtrAllocator", %"struct.llvm::Recycler<llvm::MachineBasicBlock,116ul,4ul>", %"struct.llvm::Recycler<llvm::MachineBasicBlock,116ul,4ul>", %"struct.llvm::ilist<llvm::MachineBasicBlock>", %"struct.llvm::DebugLocTracker" }
	%"struct.llvm::MachineInstr" = type { %"struct.llvm::ilist_node<llvm::MachineInstr>", %"struct.llvm::TargetInstrDesc"*, i16, %"struct.std::vector<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >", %"struct.std::list<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >", %"struct.llvm::MachineBasicBlock"*, %"struct..1$_60" }
	%"struct.llvm::MachineJumpTableInfo" = type opaque
	%"struct.llvm::MachineLocation" = type { i8, i32, i32 }
	%"struct.llvm::MachineModuleInfo" = type { %"struct.llvm::ImmutablePass", %"struct.std::vector<int,std::allocator<int> >", %"struct.std::vector<llvm::MachineMove,std::allocator<llvm::MachineMove> >", %"struct.std::vector<llvm::LandingPadInfo,std::allocator<llvm::LandingPadInfo> >", %"struct.std::vector<llvm::GlobalVariable*,std::allocator<llvm::GlobalVariable*> >", %"struct.std::vector<int,std::allocator<int> >", %"struct.std::vector<int,std::allocator<int> >", %"struct.std::vector<llvm::Function*,std::allocator<llvm::Function*> >", %"struct.llvm::SmallPtrSet<const llvm::Function*,32u>", i8, i8, i8 }
	%"struct.llvm::MachineMove" = type { i32, %"struct.llvm::MachineLocation", %"struct.llvm::MachineLocation" }
	%"struct.llvm::MachineOperand" = type { i8, i8, i8, %"struct.llvm::MachineInstr"*, %"struct.llvm::MachineOperand::$_57" }
	%"struct.llvm::MachineOperand::$_57" = type { %"struct..0$_58" }
	%"struct.llvm::MachineRegisterInfo" = type { %"struct.std::vector<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >", %"struct.std::vector<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >", %"struct.llvm::MachineOperand"**, %"struct.llvm::BitVector", %"struct.std::vector<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >", %"struct.std::vector<int,std::allocator<int> >" }
	%"struct.llvm::Module" = type opaque
	%"struct.llvm::ModulePass" = type { %"struct.llvm::Pass" }
	%"struct.llvm::PATypeHandle" = type { %"struct.llvm::Type"*, %"struct.llvm::AbstractTypeUser"* }
	%"struct.llvm::PATypeHolder" = type { %"struct.llvm::Type"* }
	%"struct.llvm::PMDataManager" = type opaque
	%"struct.llvm::Pass" = type { i32 (...)**, %"struct.llvm::AnalysisResolver"*, i32, %"struct.std::vector<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" }
	%"struct.llvm::PassInfo" = type { i8*, i8*, i32, i8, i8, i8, %"struct.std::vector<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >", %"struct.llvm::Pass"* ()* }
	%"struct.llvm::Recycler<llvm::MachineBasicBlock,116ul,4ul>" = type { %"struct.llvm::iplist<llvm::RecyclerStruct,llvm::ilist_traits<llvm::RecyclerStruct> >" }
	%"struct.llvm::RecyclerStruct" = type { %"struct.llvm::RecyclerStruct"*, %"struct.llvm::RecyclerStruct"* }
	%"struct.llvm::RecyclingAllocator<llvm::BumpPtrAllocator,llvm::SDNode,132ul,4ul>" = type { %"struct.llvm::Recycler<llvm::MachineBasicBlock,116ul,4ul>", %"struct.llvm::BumpPtrAllocator" }
	%"struct.llvm::SDNode" = type { %"struct.llvm::BumpPtrAllocator", %"struct.llvm::ilist_node<llvm::SDNode>", i16, i16, i32, %"struct.llvm::SDUse"*, %"struct.llvm::MVT"*, %"struct.llvm::SDUse"*, i16, i16, %"struct..1$_60" }
	%"struct.llvm::SDUse" = type { %"struct.llvm::SDValue", %"struct.llvm::SDNode"*, %"struct.llvm::SDUse"**, %"struct.llvm::SDUse"* }
	%"struct.llvm::SDVTList" = type { %"struct.llvm::MVT"*, i16 }
	%"struct.llvm::SDValue" = type { %"struct.llvm::SDNode"*, i32 }
	%"struct.llvm::SelectionDAG" = type { %"struct.llvm::TargetLowering"*, %"struct.llvm::MachineFunction"*, %"struct.llvm::FunctionLoweringInfo"*, %"struct.llvm::MachineModuleInfo"*, %"struct.llvm::DwarfWriter"*, %"struct.llvm::SDNode", %"struct.llvm::SDValue", %"struct.llvm::ilist<llvm::SDNode>", %"struct.llvm::RecyclingAllocator<llvm::BumpPtrAllocator,llvm::SDNode,132ul,4ul>", %"struct.llvm::FoldingSet<llvm::SDNode>", %"struct.llvm::BumpPtrAllocator", %"struct.llvm::BumpPtrAllocator", %"struct.std::map<const llvm::SDNode*,std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >", %"struct.std::vector<llvm::SDVTList,std::allocator<llvm::SDVTList> >", %"struct.std::vector<llvm::CondCodeSDNode*,std::allocator<llvm::CondCodeSDNode*> >", %"struct.std::vector<llvm::SDNode*,std::allocator<llvm::SDNode*> >", %"struct.std::map<const llvm::SDNode*,std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >", %"struct.llvm::StringMap<llvm::SDNode*,llvm::MallocAllocator>", %"struct.llvm::StringMap<llvm::SDNode*,llvm::MallocAllocator>" }
	%"struct.llvm::SmallPtrSet<const llvm::Function*,32u>" = type { %"struct.llvm::SmallPtrSetImpl", [32 x i8*] }
	%"struct.llvm::SmallPtrSetImpl" = type { i8**, i32, i32, i32, [1 x i8*] }
	%"struct.llvm::SmallVector<llvm::SDValue,16u>" = type <{ [17 x i8], [127 x i8] }>
	%"struct.llvm::SmallVector<unsigned int,1u>" = type <{ [17 x i8], [3 x i8], [3 x i32] }>
	%"struct.llvm::StringMap<llvm::SDNode*,llvm::MallocAllocator>" = type { %"struct.llvm::StringMapImpl", %struct.__false_type }
	%"struct.llvm::StringMapImpl" = type { %"struct.llvm::StringMapImpl::ItemBucket"*, i32, i32, i32, i32 }
	%"struct.llvm::StringMapImpl::ItemBucket" = type { i32, %"struct..1$_60"* }
	%"struct.llvm::TargetAsmInfo" = type opaque
	%"struct.llvm::TargetData" = type <{ %"struct.llvm::ImmutablePass", i8, i8, i8, i8, [4 x i8], %"struct.llvm::SmallVector<llvm::SDValue,16u>" }>
	%"struct.llvm::TargetFrameInfo" = type { i32 (...)**, i32, i32, i32 }
	%"struct.llvm::TargetInstrDesc" = type { i16, i16, i16, i16, i8*, i32, i32, i32*, i32*, %"struct.llvm::TargetRegisterClass"**, %"struct.llvm::TargetOperandInfo"* }
	%"struct.llvm::TargetLowering" = type { i32 (...)**, %"struct.llvm::TargetMachine"*, %"struct.llvm::TargetData"*, %"struct.llvm::MVT", i8, i8, i8, i8, i8, i8, i8, %"struct.llvm::MVT", i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [30 x %"struct.llvm::TargetRegisterClass"*], [30 x i8], [30 x %"struct.llvm::MVT"], [30 x %"struct.llvm::MVT"], [179 x i64], [4 x i64], [30 x i64], [2 x [5 x i64]], [30 x i64], [24 x i64], %"struct.llvm::TargetLowering::ValueTypeActionImpl", %"struct.std::vector<llvm::APFloat,std::allocator<llvm::APFloat> >", %"struct.std::vector<std::pair<llvm::MVT, llvm::TargetRegisterClass*>,std::allocator<std::pair<llvm::MVT, llvm::TargetRegisterClass*> > >", [23 x i8], %"struct.std::map<const llvm::SDNode*,std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >", [180 x i8*], [180 x i32], i32, i32, i32, i8 }
	%"struct.llvm::TargetLowering::ValueTypeActionImpl" = type { [2 x i32] }
	%"struct.llvm::TargetMachine" = type { i32 (...)**, %"struct.llvm::TargetAsmInfo"* }
	%"struct.llvm::TargetOperandInfo" = type { i16, i16, i32 }
	%"struct.llvm::TargetRegisterClass" = type { i32 (...)**, i32, i8, %"struct.llvm::MVT"*, %"struct.llvm::TargetRegisterClass"**, %"struct.llvm::TargetRegisterClass"**, i32, i32, i32, i32*, i32* }
	%"struct.llvm::Type" = type { %"struct.llvm::AbstractTypeUser", i8, [3 x i8], i32, %"struct.llvm::Type"*, %"struct.std::vector<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >", i32, %"struct.llvm::PATypeHandle"* }
	%"struct.llvm::Use" = type { %"struct.llvm::Value"*, %"struct.llvm::Use"*, %"struct..1$_60" }
	%"struct.llvm::User" = type { %"struct.llvm::Value", %"struct.llvm::Use"*, i32 }
	%"struct.llvm::Value" = type { i32 (...)**, i16, i16, %"struct.llvm::PATypeHolder", %"struct.llvm::Use"*, %"struct.llvm::ValueName"* }
	%"struct.llvm::ValueName" = type opaque
	%"struct.llvm::ValueSymbolTable" = type opaque
	%"struct.llvm::fltSemantics" = type opaque
	%"struct.llvm::ilist<llvm::MachineBasicBlock>" = type { %"struct.llvm::iplist<llvm::MachineBasicBlock,llvm::ilist_traits<llvm::MachineBasicBlock> >" }
	%"struct.llvm::ilist<llvm::MachineInstr>" = type { %"struct.llvm::iplist<llvm::MachineInstr,llvm::ilist_traits<llvm::MachineInstr> >" }
	%"struct.llvm::ilist<llvm::SDNode>" = type { %"struct.llvm::iplist<llvm::SDNode,llvm::ilist_traits<llvm::SDNode> >" }
	%"struct.llvm::ilist_node<llvm::Argument>" = type { %"struct.llvm::Argument"*, %"struct.llvm::Argument"* }
	%"struct.llvm::ilist_node<llvm::BasicBlock>" = type { %"struct.llvm::BasicBlock"*, %"struct.llvm::BasicBlock"* }
	%"struct.llvm::ilist_node<llvm::Function>" = type { %"struct.llvm::Function"*, %"struct.llvm::Function"* }
	%"struct.llvm::ilist_node<llvm::GlobalVariable>" = type { %"struct.llvm::GlobalVariable"*, %"struct.llvm::GlobalVariable"* }
	%"struct.llvm::ilist_node<llvm::Instruction>" = type { %"struct.llvm::Instruction"*, %"struct.llvm::Instruction"* }
	%"struct.llvm::ilist_node<llvm::MachineBasicBlock>" = type { %"struct.llvm::MachineBasicBlock"*, %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::ilist_node<llvm::MachineInstr>" = type { %"struct.llvm::MachineInstr"*, %"struct.llvm::MachineInstr"* }
	%"struct.llvm::ilist_node<llvm::SDNode>" = type { %"struct.llvm::SDNode"*, %"struct.llvm::SDNode"* }
	%"struct.llvm::ilist_traits<llvm::MachineBasicBlock>" = type { %"struct.llvm::MachineBasicBlock" }
	%"struct.llvm::ilist_traits<llvm::MachineInstr>" = type { %"struct.llvm::MachineInstr", %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::ilist_traits<llvm::RecyclerStruct>" = type { %"struct.llvm::RecyclerStruct" }
	%"struct.llvm::ilist_traits<llvm::SDNode>" = type { %"struct.llvm::SDNode" }
	%"struct.llvm::iplist<llvm::Argument,llvm::ilist_traits<llvm::Argument> >" = type { %"struct.llvm::Argument"* }
	%"struct.llvm::iplist<llvm::BasicBlock,llvm::ilist_traits<llvm::BasicBlock> >" = type { %"struct.llvm::BasicBlock"* }
	%"struct.llvm::iplist<llvm::Instruction,llvm::ilist_traits<llvm::Instruction> >" = type { %"struct.llvm::Instruction"* }
	%"struct.llvm::iplist<llvm::MachineBasicBlock,llvm::ilist_traits<llvm::MachineBasicBlock> >" = type { %"struct.llvm::ilist_traits<llvm::MachineBasicBlock>", %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::iplist<llvm::MachineInstr,llvm::ilist_traits<llvm::MachineInstr> >" = type { %"struct.llvm::ilist_traits<llvm::MachineInstr>", %"struct.llvm::MachineInstr"* }
	%"struct.llvm::iplist<llvm::RecyclerStruct,llvm::ilist_traits<llvm::RecyclerStruct> >" = type { %"struct.llvm::ilist_traits<llvm::RecyclerStruct>", %"struct.llvm::RecyclerStruct"* }
	%"struct.llvm::iplist<llvm::SDNode,llvm::ilist_traits<llvm::SDNode> >" = type { %"struct.llvm::ilist_traits<llvm::SDNode>", %"struct.llvm::SDNode"* }
	%"struct.std::_List_base<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >" = type { %"struct.llvm::ilist_traits<llvm::RecyclerStruct>" }
	%"struct.std::_Rb_tree<const llvm::SDNode*,std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >,std::_Select1st<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >" = type { %"struct.std::_Rb_tree<const llvm::SDNode*,std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >,std::_Select1st<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >::_Rb_tree_impl<std::less<const llvm::SDNode*>,false>" }
	%"struct.std::_Rb_tree<const llvm::SDNode*,std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >,std::_Select1st<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >::_Rb_tree_impl<std::less<const llvm::SDNode*>,false>" = type { %struct.__false_type, %"struct.std::_Rb_tree_node_base", i32 }
	%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
	%"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" = type { %"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >::_Vector_impl" }
	%"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >::_Vector_impl" = type { %"struct.llvm::PassInfo"**, %"struct.llvm::PassInfo"**, %"struct.llvm::PassInfo"** }
	%"struct.std::_Vector_base<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" }
	%"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" = type { i32*, i32*, i32* }
	%"struct.std::_Vector_base<llvm::APFloat,std::allocator<llvm::APFloat> >" = type { %"struct.std::_Vector_base<llvm::APFloat,std::allocator<llvm::APFloat> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::APFloat,std::allocator<llvm::APFloat> >::_Vector_impl" = type { %"struct.llvm::APFloat"*, %"struct.llvm::APFloat"*, %"struct.llvm::APFloat"* }
	%"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" = type { %"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >::_Vector_impl" = type { %"struct.llvm::AbstractTypeUser"**, %"struct.llvm::AbstractTypeUser"**, %"struct.llvm::AbstractTypeUser"** }
	%"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >" = type { %"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >::_Vector_impl" = type { %"struct.llvm::CalleeSavedInfo"*, %"struct.llvm::CalleeSavedInfo"*, %"struct.llvm::CalleeSavedInfo"* }
	%"struct.std::_Vector_base<llvm::CondCodeSDNode*,std::allocator<llvm::CondCodeSDNode*> >" = type { %"struct.std::_Vector_base<llvm::CondCodeSDNode*,std::allocator<llvm::CondCodeSDNode*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::CondCodeSDNode*,std::allocator<llvm::CondCodeSDNode*> >::_Vector_impl" = type { %"struct.llvm::CondCodeSDNode"**, %"struct.llvm::CondCodeSDNode"**, %"struct.llvm::CondCodeSDNode"** }
	%"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >" = type { %"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >::_Vector_impl" = type { %"struct.llvm::DebugLocTuple"*, %"struct.llvm::DebugLocTuple"*, %"struct.llvm::DebugLocTuple"* }
	%"struct.std::_Vector_base<llvm::Function*,std::allocator<llvm::Function*> >" = type { %"struct.std::_Vector_base<llvm::Function*,std::allocator<llvm::Function*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::Function*,std::allocator<llvm::Function*> >::_Vector_impl" = type { %"struct.llvm::Function"**, %"struct.llvm::Function"**, %"struct.llvm::Function"** }
	%"struct.std::_Vector_base<llvm::GlobalVariable*,std::allocator<llvm::GlobalVariable*> >" = type { %"struct.std::_Vector_base<llvm::GlobalVariable*,std::allocator<llvm::GlobalVariable*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::GlobalVariable*,std::allocator<llvm::GlobalVariable*> >::_Vector_impl" = type { %"struct.llvm::GlobalVariable"**, %"struct.llvm::GlobalVariable"**, %"struct.llvm::GlobalVariable"** }
	%"struct.std::_Vector_base<llvm::LandingPadInfo,std::allocator<llvm::LandingPadInfo> >" = type { %"struct.std::_Vector_base<llvm::LandingPadInfo,std::allocator<llvm::LandingPadInfo> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::LandingPadInfo,std::allocator<llvm::LandingPadInfo> >::_Vector_impl" = type { %"struct.llvm::LandingPadInfo"*, %"struct.llvm::LandingPadInfo"*, %"struct.llvm::LandingPadInfo"* }
	%"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >" = type { %"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >::_Vector_impl" = type { %"struct.llvm::MachineBasicBlock"**, %"struct.llvm::MachineBasicBlock"**, %"struct.llvm::MachineBasicBlock"** }
	%"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >" = type { %"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >::_Vector_impl" = type { %"struct.llvm::MachineFrameInfo::StackObject"*, %"struct.llvm::MachineFrameInfo::StackObject"*, %"struct.llvm::MachineFrameInfo::StackObject"* }
	%"struct.std::_Vector_base<llvm::MachineMove,std::allocator<llvm::MachineMove> >" = type { %"struct.std::_Vector_base<llvm::MachineMove,std::allocator<llvm::MachineMove> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineMove,std::allocator<llvm::MachineMove> >::_Vector_impl" = type { %"struct.llvm::MachineMove"*, %"struct.llvm::MachineMove"*, %"struct.llvm::MachineMove"* }
	%"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" = type { %"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >::_Vector_impl" = type { %"struct.llvm::MachineOperand"*, %"struct.llvm::MachineOperand"*, %"struct.llvm::MachineOperand"* }
	%"struct.std::_Vector_base<llvm::SDNode*,std::allocator<llvm::SDNode*> >" = type { %"struct.std::_Vector_base<llvm::SDNode*,std::allocator<llvm::SDNode*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::SDNode*,std::allocator<llvm::SDNode*> >::_Vector_impl" = type { %"struct.llvm::SDNode"**, %"struct.llvm::SDNode"**, %"struct.llvm::SDNode"** }
	%"struct.std::_Vector_base<llvm::SDVTList,std::allocator<llvm::SDVTList> >" = type { %"struct.std::_Vector_base<llvm::SDVTList,std::allocator<llvm::SDVTList> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::SDVTList,std::allocator<llvm::SDVTList> >::_Vector_impl" = type { %"struct.llvm::SDVTList"*, %"struct.llvm::SDVTList"*, %"struct.llvm::SDVTList"* }
	%"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >::_Vector_impl" = type { %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"*, %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"*, %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"* }
	%"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >::_Vector_impl" = type { %"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>"*, %"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>"*, %"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>"* }
	%"struct.std::_Vector_base<std::pair<llvm::MVT, llvm::TargetRegisterClass*>,std::allocator<std::pair<llvm::MVT, llvm::TargetRegisterClass*> > >" = type { %"struct.std::_Vector_base<std::pair<llvm::MVT, llvm::TargetRegisterClass*>,std::allocator<std::pair<llvm::MVT, llvm::TargetRegisterClass*> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<llvm::MVT, llvm::TargetRegisterClass*>,std::allocator<std::pair<llvm::MVT, llvm::TargetRegisterClass*> > >::_Vector_impl" = type { %"struct.std::pair<llvm::MVT,llvm::TargetRegisterClass*>"*, %"struct.std::pair<llvm::MVT,llvm::TargetRegisterClass*>"*, %"struct.std::pair<llvm::MVT,llvm::TargetRegisterClass*>"* }
	%"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >" = type { %"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" = type { %"struct.std::pair<int,int>"*, %"struct.std::pair<int,int>"*, %"struct.std::pair<int,int>"* }
	%"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >" = type { %"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >::_Vector_impl" = type { %"struct.std::vector<int,std::allocator<int> >"*, %"struct.std::vector<int,std::allocator<int> >"*, %"struct.std::vector<int,std::allocator<int> >"* }
	%"struct.std::list<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >" = type { %"struct.std::_List_base<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >" }
	%"struct.std::map<const llvm::SDNode*,std::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >" = type { %"struct.std::_Rb_tree<const llvm::SDNode*,std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >,std::_Select1st<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >,std::less<const llvm::SDNode*>,std::allocator<std::pair<const llvm::SDNode* const, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >" }
	%"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>" = type { %"struct.llvm::PassInfo"*, %"struct.llvm::Pass"* }
	%"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>" = type { %"struct.llvm::TargetRegisterClass"*, %"struct.llvm::MachineOperand"* }
	%"struct.std::pair<int,int>" = type { i32, i32 }
	%"struct.std::pair<llvm::DebugLocTuple,unsigned int>" = type { %"struct.llvm::DebugLocTuple", i32 }
	%"struct.std::pair<llvm::MVT,llvm::TargetRegisterClass*>" = type { %"struct.llvm::MVT", %"struct.llvm::TargetRegisterClass"* }
	%"struct.std::string" = type { %"struct.llvm::BumpPtrAllocator" }
	%"struct.std::vector<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" = type { %"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" }
	%"struct.std::vector<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >" }
	%"struct.std::vector<llvm::APFloat,std::allocator<llvm::APFloat> >" = type { %"struct.std::_Vector_base<llvm::APFloat,std::allocator<llvm::APFloat> >" }
	%"struct.std::vector<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" = type { %"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" }
	%"struct.std::vector<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >" = type { %"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >" }
	%"struct.std::vector<llvm::CondCodeSDNode*,std::allocator<llvm::CondCodeSDNode*> >" = type { %"struct.std::_Vector_base<llvm::CondCodeSDNode*,std::allocator<llvm::CondCodeSDNode*> >" }
	%"struct.std::vector<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >" = type { %"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >" }
	%"struct.std::vector<llvm::Function*,std::allocator<llvm::Function*> >" = type { %"struct.std::_Vector_base<llvm::Function*,std::allocator<llvm::Function*> >" }
	%"struct.std::vector<llvm::GlobalVariable*,std::allocator<llvm::GlobalVariable*> >" = type { %"struct.std::_Vector_base<llvm::GlobalVariable*,std::allocator<llvm::GlobalVariable*> >" }
	%"struct.std::vector<llvm::LandingPadInfo,std::allocator<llvm::LandingPadInfo> >" = type { %"struct.std::_Vector_base<llvm::LandingPadInfo,std::allocator<llvm::LandingPadInfo> >" }
	%"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >" = type { %"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >" }
	%"struct.std::vector<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >" = type { %"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >" }
	%"struct.std::vector<llvm::MachineMove,std::allocator<llvm::MachineMove> >" = type { %"struct.std::_Vector_base<llvm::MachineMove,std::allocator<llvm::MachineMove> >" }
	%"struct.std::vector<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" = type { %"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" }
	%"struct.std::vector<llvm::SDNode*,std::allocator<llvm::SDNode*> >" = type { %"struct.std::_Vector_base<llvm::SDNode*,std::allocator<llvm::SDNode*> >" }
	%"struct.std::vector<llvm::SDVTList,std::allocator<llvm::SDVTList> >" = type { %"struct.std::_Vector_base<llvm::SDVTList,std::allocator<llvm::SDVTList> >" }
	%"struct.std::vector<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" }
	%"struct.std::vector<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >" }
	%"struct.std::vector<std::pair<llvm::MVT, llvm::TargetRegisterClass*>,std::allocator<std::pair<llvm::MVT, llvm::TargetRegisterClass*> > >" = type { %"struct.std::_Vector_base<std::pair<llvm::MVT, llvm::TargetRegisterClass*>,std::allocator<std::pair<llvm::MVT, llvm::TargetRegisterClass*> > >" }
	%"struct.std::vector<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >" = type { %"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >" }
	%"struct.std::vector<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >" = type { %"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >" }
@"\01LC81" = internal constant [65 x i8] c"/Users/echeng/LLVM/llvm/include/llvm/CodeGen/SelectionDAGNodes.h\00"		; <[65 x i8]*> [#uses=1]
@_ZZNK4llvm6SDNode12getValueTypeEjE8__func__ = internal constant [13 x i8] c"getValueType\00"		; <[13 x i8]*> [#uses=1]
@"\01LC83" = internal constant [46 x i8] c"ResNo < NumValues && \22Illegal result number!\22\00"		; <[46 x i8]*> [#uses=1]
@"\01LC197" = internal constant [16 x i8] c"___tls_get_addr\00"		; <[16 x i8]*> [#uses=1]
@llvm.used1 = appending global [1 x i8*] [ i8* bitcast (i64 (%"struct.llvm::GlobalAddressSDNode"*, %"struct.llvm::SelectionDAG"*, %"struct.llvm::MVT"*)* @_ZL31LowerToTLSGeneralDynamicModel32PN4llvm19GlobalAddressSDNodeERNS_12SelectionDAGENS_3MVTE to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define fastcc i64 @_ZL31LowerToTLSGeneralDynamicModel32PN4llvm19GlobalAddressSDNodeERNS_12SelectionDAGENS_3MVTE(%"struct.llvm::GlobalAddressSDNode"* %GA, %"struct.llvm::SelectionDAG"* %DAG, %"struct.llvm::MVT"* byval align 4 %PtrVT) nounwind noinline {
entry:
	%VT2.i185 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%VT1.i186 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%Ops.i187 = alloca [4 x %"struct.llvm::SDValue"], align 8		; <[4 x %"struct.llvm::SDValue"]*> [#uses=9]
	%0 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%VT182 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%VT2.i173 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%VT1.i174 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%Ops.i175 = alloca [4 x %"struct.llvm::SDValue"], align 8		; <[4 x %"struct.llvm::SDValue"]*> [#uses=9]
	%1 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%VT3.i = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%VT2.i = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%VT1.i = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%Ops.i = alloca [3 x %"struct.llvm::SDValue"], align 8		; <[3 x %"struct.llvm::SDValue"]*> [#uses=7]
	%VT = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%Ops1 = alloca [5 x %"struct.llvm::SDValue"], align 8		; <[5 x %"struct.llvm::SDValue"]*> [#uses=11]
	%Ops = alloca [3 x %"struct.llvm::SDValue"], align 8		; <[3 x %"struct.llvm::SDValue"]*> [#uses=7]
	%NodeTys = alloca %"struct.llvm::SDVTList", align 8		; <%"struct.llvm::SDVTList"*> [#uses=4]
	%2 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%3 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%4 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%5 = alloca %"struct.llvm::MVT", align 8		; <%"struct.llvm::MVT"*> [#uses=2]
	%6 = getelementptr %"struct.llvm::GlobalAddressSDNode"* %GA, i32 0, i32 0, i32 10, i32 0		; <i32*> [#uses=1]
	%7 = load i32* %6, align 4		; <i32> [#uses=5]
	%8 = call i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocENS_3MVTE(%"struct.llvm::SelectionDAG"* %DAG, i32 208, i32 0, %"struct.llvm::MVT"* byval align 4 %PtrVT) nounwind		; <i64> [#uses=2]
	%9 = trunc i64 %8 to i32		; <i32> [#uses=1]
	%sroa.store.elt = lshr i64 %8, 32		; <i64> [#uses=1]
	%10 = trunc i64 %sroa.store.elt to i32		; <i32> [#uses=3]
	%tmp52 = inttoptr i32 %9 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=3]
	%11 = getelementptr %"struct.llvm::SelectionDAG"* %DAG, i32 0, i32 5		; <%"struct.llvm::SDNode"*> [#uses=1]
	%12 = getelementptr %"struct.llvm::MVT"* %VT1.i186, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %12, align 8
	%13 = getelementptr %"struct.llvm::MVT"* %VT2.i185, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 12, i32* %13, align 8
	%14 = call i64 @_ZN4llvm12SelectionDAG9getVTListENS_3MVTES1_(%"struct.llvm::SelectionDAG"* %DAG, %"struct.llvm::MVT"* byval align 4 %VT1.i186, %"struct.llvm::MVT"* byval align 4 %VT2.i185) nounwind		; <i64> [#uses=1]
	%15 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 0, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %11, %"struct.llvm::SDNode"** %15, align 8
	%16 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 0, i32* %16, align 4
	%17 = getelementptr %"struct.llvm::SDNode"* %tmp52, i32 0, i32 9		; <i16*> [#uses=1]
	%18 = load i16* %17, align 2		; <i16> [#uses=1]
	%19 = zext i16 %18 to i32		; <i32> [#uses=1]
	%20 = icmp ugt i32 %19, %10		; <i1> [#uses=1]
	br i1 %20, label %_ZN4llvm12SelectionDAG12getCopyToRegENS_7SDValueENS_8DebugLocEjS1_S1_.exit193, label %bb.i.i.i188

bb.i.i.i188:		; preds = %entry
	call void @__assert_rtn(i8* getelementptr ([13 x i8]* @_ZZNK4llvm6SDNode12getValueTypeEjE8__func__, i32 0, i32 0), i8* getelementptr ([65 x i8]* @"\01LC81", i32 0, i32 0), i32 1314, i8* getelementptr ([46 x i8]* @"\01LC83", i32 0, i32 0)) noreturn nounwind
	unreachable

_ZN4llvm12SelectionDAG12getCopyToRegENS_7SDValueENS_8DebugLocEjS1_S1_.exit193:		; preds = %entry
	%21 = trunc i64 %14 to i32		; <i32> [#uses=1]
	%tmp4.i.i189 = inttoptr i32 %21 to %"struct.llvm::MVT"*		; <%"struct.llvm::MVT"*> [#uses=1]
	%22 = getelementptr %"struct.llvm::SDNode"* %tmp52, i32 0, i32 6		; <%"struct.llvm::MVT"**> [#uses=1]
	%23 = load %"struct.llvm::MVT"** %22, align 4		; <%"struct.llvm::MVT"*> [#uses=1]
	%24 = getelementptr %"struct.llvm::MVT"* %23, i32 %10, i32 0, i32 0		; <i32*> [#uses=1]
	%25 = load i32* %24, align 4		; <i32> [#uses=1]
	%26 = getelementptr %"struct.llvm::MVT"* %0, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %25, i32* %26, align 8
	%27 = call i64 @_ZN4llvm12SelectionDAG11getRegisterEjNS_3MVTE(%"struct.llvm::SelectionDAG"* %DAG, i32 19, %"struct.llvm::MVT"* byval align 4 %0) nounwind		; <i64> [#uses=2]
	%28 = trunc i64 %27 to i32		; <i32> [#uses=1]
	%sroa.store.elt.i190 = lshr i64 %27, 32		; <i64> [#uses=1]
	%29 = trunc i64 %sroa.store.elt.i190 to i32		; <i32> [#uses=1]
	%30 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 1, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	%tmp5.i191 = inttoptr i32 %28 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp5.i191, %"struct.llvm::SDNode"** %30, align 8
	%31 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	store i32 %29, i32* %31, align 4
	%32 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 2, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp52, %"struct.llvm::SDNode"** %32, align 8
	%33 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	store i32 %10, i32* %33, align 4
	%34 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 3, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* null, %"struct.llvm::SDNode"** %34, align 8
	%35 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 3, i32 1		; <i32*> [#uses=1]
	store i32 0, i32* %35, align 4
	%36 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i187, i32 0, i32 0		; <%"struct.llvm::SDValue"*> [#uses=1]
	%37 = call i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocEPKNS_3MVTEjPKNS_7SDValueEj(%"struct.llvm::SelectionDAG"* %DAG, i32 36, i32 %7, %"struct.llvm::MVT"* %tmp4.i.i189, i32 2, %"struct.llvm::SDValue"* %36, i32 3) nounwind		; <i64> [#uses=2]
	%38 = trunc i64 %37 to i32		; <i32> [#uses=1]
	%tmp66 = inttoptr i32 %38 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=2]
	%39 = getelementptr %"struct.llvm::MVT"* %5, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 12, i32* %39, align 8
	%40 = getelementptr %"struct.llvm::MVT"* %4, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %40, align 8
	%41 = call i64 @_ZN4llvm12SelectionDAG9getVTListENS_3MVTES1_S1_(%"struct.llvm::SelectionDAG"* %DAG, %"struct.llvm::MVT"* byval align 4 %PtrVT, %"struct.llvm::MVT"* byval align 4 %4, %"struct.llvm::MVT"* byval align 4 %5) nounwind		; <i64> [#uses=2]
	%42 = trunc i64 %41 to i32		; <i32> [#uses=1]
	%sroa.store.elt75 = lshr i64 %41, 32		; <i64> [#uses=1]
	%43 = trunc i64 %sroa.store.elt75 to i16		; <i16> [#uses=1]
	%44 = getelementptr %"struct.llvm::SDVTList"* %NodeTys, i32 0, i32 0		; <%"struct.llvm::MVT"**> [#uses=2]
	%tmp78 = inttoptr i32 %42 to %"struct.llvm::MVT"*		; <%"struct.llvm::MVT"*> [#uses=1]
	store %"struct.llvm::MVT"* %tmp78, %"struct.llvm::MVT"** %44, align 8
	%45 = getelementptr %"struct.llvm::SDVTList"* %NodeTys, i32 0, i32 1		; <i16*> [#uses=2]
	store i16 %43, i16* %45, align 4
	%46 = getelementptr %"struct.llvm::GlobalAddressSDNode"* %GA, i32 0, i32 0, i32 9		; <i16*> [#uses=1]
	%47 = load i16* %46, align 2		; <i16> [#uses=1]
	%48 = icmp eq i16 %47, 0		; <i1> [#uses=1]
	br i1 %48, label %bb.i, label %_ZNK4llvm6SDNode12getValueTypeEj.exit

bb.i:		; preds = %_ZN4llvm12SelectionDAG12getCopyToRegENS_7SDValueENS_8DebugLocEjS1_S1_.exit193
	call void @__assert_rtn(i8* getelementptr ([13 x i8]* @_ZZNK4llvm6SDNode12getValueTypeEjE8__func__, i32 0, i32 0), i8* getelementptr ([65 x i8]* @"\01LC81", i32 0, i32 0), i32 1314, i8* getelementptr ([46 x i8]* @"\01LC83", i32 0, i32 0)) noreturn nounwind
	unreachable

_ZNK4llvm6SDNode12getValueTypeEj.exit:		; preds = %_ZN4llvm12SelectionDAG12getCopyToRegENS_7SDValueENS_8DebugLocEjS1_S1_.exit193
	%sroa.store.elt63 = lshr i64 %37, 32		; <i64> [#uses=1]
	%49 = trunc i64 %sroa.store.elt63 to i32		; <i32> [#uses=1]
	%50 = getelementptr %"struct.llvm::GlobalAddressSDNode"* %GA, i32 0, i32 2		; <i64*> [#uses=1]
	%51 = load i64* %50, align 4		; <i64> [#uses=1]
	%52 = getelementptr %"struct.llvm::GlobalAddressSDNode"* %GA, i32 0, i32 0, i32 6		; <%"struct.llvm::MVT"**> [#uses=1]
	%53 = load %"struct.llvm::MVT"** %52, align 4		; <%"struct.llvm::MVT"*> [#uses=1]
	%54 = getelementptr %"struct.llvm::MVT"* %53, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%55 = load i32* %54, align 4		; <i32> [#uses=1]
	%56 = getelementptr %"struct.llvm::GlobalAddressSDNode"* %GA, i32 0, i32 1		; <%"struct.llvm::GlobalValue"**> [#uses=1]
	%57 = load %"struct.llvm::GlobalValue"** %56, align 4		; <%"struct.llvm::GlobalValue"*> [#uses=1]
	%58 = getelementptr %"struct.llvm::MVT"* %VT182, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %55, i32* %58, align 8
	%59 = call i64 @_ZN4llvm12SelectionDAG16getGlobalAddressEPKNS_11GlobalValueENS_3MVTExb(%"struct.llvm::SelectionDAG"* %DAG, %"struct.llvm::GlobalValue"* %57, %"struct.llvm::MVT"* byval align 4 %VT182, i64 %51, i8 zeroext 1) nounwind		; <i64> [#uses=2]
	%60 = trunc i64 %59 to i32		; <i32> [#uses=1]
	%sroa.store.elt83 = lshr i64 %59, 32		; <i64> [#uses=1]
	%61 = trunc i64 %sroa.store.elt83 to i32		; <i32> [#uses=1]
	%tmp86 = inttoptr i32 %60 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=1]
	%62 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops, i32 0, i32 0, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp66, %"struct.llvm::SDNode"** %62, align 8
	%63 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %49, i32* %63, align 4
	%64 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops, i32 0, i32 1, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp86, %"struct.llvm::SDNode"** %64, align 8
	%65 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	store i32 %61, i32* %65, align 4
	%66 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops, i32 0, i32 2, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp66, %"struct.llvm::SDNode"** %66, align 8
	%67 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	store i32 1, i32* %67, align 4
	%68 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops, i32 0, i32 0		; <%"struct.llvm::SDValue"*> [#uses=1]
	%69 = call i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocENS_8SDVTListEPKNS_7SDValueEj(%"struct.llvm::SelectionDAG"* %DAG, i32 220, i32 %7, %"struct.llvm::SDVTList"* byval align 4 %NodeTys, %"struct.llvm::SDValue"* %68, i32 3) nounwind		; <i64> [#uses=2]
	%70 = trunc i64 %69 to i32		; <i32> [#uses=1]
	%sroa.store.elt89 = lshr i64 %69, 32		; <i64> [#uses=1]
	%71 = trunc i64 %sroa.store.elt89 to i32		; <i32> [#uses=3]
	%tmp92 = inttoptr i32 %70 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=7]
	call void @_ZNK4llvm6SDNode4dumpEv(%"struct.llvm::SDNode"* %tmp92) nounwind
	%72 = getelementptr %"struct.llvm::MVT"* %VT1.i174, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %72, align 8
	%73 = getelementptr %"struct.llvm::MVT"* %VT2.i173, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 12, i32* %73, align 8
	%74 = call i64 @_ZN4llvm12SelectionDAG9getVTListENS_3MVTES1_(%"struct.llvm::SelectionDAG"* %DAG, %"struct.llvm::MVT"* byval align 4 %VT1.i174, %"struct.llvm::MVT"* byval align 4 %VT2.i173) nounwind		; <i64> [#uses=1]
	%75 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 0, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp92, %"struct.llvm::SDNode"** %75, align 8
	%76 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 1, i32* %76, align 4
	%77 = getelementptr %"struct.llvm::SDNode"* %tmp92, i32 0, i32 9		; <i16*> [#uses=1]
	%78 = load i16* %77, align 2		; <i16> [#uses=1]
	%79 = zext i16 %78 to i32		; <i32> [#uses=1]
	%80 = icmp ugt i32 %79, %71		; <i1> [#uses=1]
	br i1 %80, label %_ZN4llvm12SelectionDAG12getCopyToRegENS_7SDValueENS_8DebugLocEjS1_S1_.exit, label %bb.i.i.i

bb.i.i.i:		; preds = %_ZNK4llvm6SDNode12getValueTypeEj.exit
	call void @__assert_rtn(i8* getelementptr ([13 x i8]* @_ZZNK4llvm6SDNode12getValueTypeEjE8__func__, i32 0, i32 0), i8* getelementptr ([65 x i8]* @"\01LC81", i32 0, i32 0), i32 1314, i8* getelementptr ([46 x i8]* @"\01LC83", i32 0, i32 0)) noreturn nounwind
	unreachable

_ZN4llvm12SelectionDAG12getCopyToRegENS_7SDValueENS_8DebugLocEjS1_S1_.exit:		; preds = %_ZNK4llvm6SDNode12getValueTypeEj.exit
	%81 = trunc i64 %74 to i32		; <i32> [#uses=1]
	%tmp4.i.i176 = inttoptr i32 %81 to %"struct.llvm::MVT"*		; <%"struct.llvm::MVT"*> [#uses=1]
	%82 = getelementptr %"struct.llvm::SDNode"* %tmp92, i32 0, i32 6		; <%"struct.llvm::MVT"**> [#uses=1]
	%83 = load %"struct.llvm::MVT"** %82, align 4		; <%"struct.llvm::MVT"*> [#uses=1]
	%84 = getelementptr %"struct.llvm::MVT"* %83, i32 %71, i32 0, i32 0		; <i32*> [#uses=1]
	%85 = load i32* %84, align 4		; <i32> [#uses=1]
	%86 = getelementptr %"struct.llvm::MVT"* %1, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %85, i32* %86, align 8
	%87 = call i64 @_ZN4llvm12SelectionDAG11getRegisterEjNS_3MVTE(%"struct.llvm::SelectionDAG"* %DAG, i32 17, %"struct.llvm::MVT"* byval align 4 %1) nounwind		; <i64> [#uses=2]
	%88 = trunc i64 %87 to i32		; <i32> [#uses=1]
	%sroa.store.elt.i177 = lshr i64 %87, 32		; <i64> [#uses=1]
	%89 = trunc i64 %sroa.store.elt.i177 to i32		; <i32> [#uses=1]
	%90 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 1, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	%tmp5.i178 = inttoptr i32 %88 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp5.i178, %"struct.llvm::SDNode"** %90, align 8
	%91 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	store i32 %89, i32* %91, align 4
	%92 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 2, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp92, %"struct.llvm::SDNode"** %92, align 8
	%93 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	store i32 %71, i32* %93, align 4
	%94 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 3, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp92, %"struct.llvm::SDNode"** %94, align 8
	%95 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 3, i32 1		; <i32*> [#uses=1]
	store i32 2, i32* %95, align 4
	%96 = icmp eq %"struct.llvm::SDNode"* %tmp92, null		; <i1> [#uses=1]
	%iftmp.583.0.i = select i1 %96, i32 3, i32 4		; <i32> [#uses=1]
	%97 = getelementptr [4 x %"struct.llvm::SDValue"]* %Ops.i175, i32 0, i32 0		; <%"struct.llvm::SDValue"*> [#uses=1]
	%98 = call i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocEPKNS_3MVTEjPKNS_7SDValueEj(%"struct.llvm::SelectionDAG"* %DAG, i32 36, i32 %7, %"struct.llvm::MVT"* %tmp4.i.i176, i32 2, %"struct.llvm::SDValue"* %97, i32 %iftmp.583.0.i) nounwind		; <i64> [#uses=2]
	%99 = trunc i64 %98 to i32		; <i32> [#uses=1]
	%sroa.store.elt107 = lshr i64 %98, 32		; <i64> [#uses=1]
	%100 = trunc i64 %sroa.store.elt107 to i32		; <i32> [#uses=1]
	%tmp110 = inttoptr i32 %99 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=2]
	%101 = getelementptr %"struct.llvm::MVT"* %3, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 12, i32* %101, align 8
	%102 = getelementptr %"struct.llvm::MVT"* %2, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %102, align 8
	%103 = call i64 @_ZN4llvm12SelectionDAG9getVTListENS_3MVTES1_(%"struct.llvm::SelectionDAG"* %DAG, %"struct.llvm::MVT"* byval align 4 %2, %"struct.llvm::MVT"* byval align 4 %3) nounwind		; <i64> [#uses=2]
	%104 = trunc i64 %103 to i32		; <i32> [#uses=1]
	%sroa.store.elt119 = lshr i64 %103, 32		; <i64> [#uses=1]
	%105 = trunc i64 %sroa.store.elt119 to i16		; <i16> [#uses=1]
	%tmp122 = inttoptr i32 %104 to %"struct.llvm::MVT"*		; <%"struct.llvm::MVT"*> [#uses=1]
	store %"struct.llvm::MVT"* %tmp122, %"struct.llvm::MVT"** %44, align 8
	store i16 %105, i16* %45, align 4
	%106 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 0, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp110, %"struct.llvm::SDNode"** %106, align 8
	%107 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %100, i32* %107, align 4
	%108 = call i64 @_ZN4llvm12SelectionDAG23getTargetExternalSymbolEPKcNS_3MVTE(%"struct.llvm::SelectionDAG"* %DAG, i8* getelementptr ([16 x i8]* @"\01LC197", i32 0, i32 0), %"struct.llvm::MVT"* byval align 4 %PtrVT) nounwind		; <i64> [#uses=2]
	%109 = trunc i64 %108 to i32		; <i32> [#uses=1]
	%sroa.store.elt125 = lshr i64 %108, 32		; <i64> [#uses=1]
	%110 = trunc i64 %sroa.store.elt125 to i32		; <i32> [#uses=1]
	%111 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 1, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	%tmp128 = inttoptr i32 %109 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp128, %"struct.llvm::SDNode"** %111, align 8
	%112 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	store i32 %110, i32* %112, align 4
	%113 = call i64 @_ZN4llvm12SelectionDAG11getRegisterEjNS_3MVTE(%"struct.llvm::SelectionDAG"* %DAG, i32 17, %"struct.llvm::MVT"* byval align 4 %PtrVT) nounwind		; <i64> [#uses=2]
	%114 = trunc i64 %113 to i32		; <i32> [#uses=1]
	%sroa.store.elt131 = lshr i64 %113, 32		; <i64> [#uses=1]
	%115 = trunc i64 %sroa.store.elt131 to i32		; <i32> [#uses=1]
	%116 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 2, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	%tmp134 = inttoptr i32 %114 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp134, %"struct.llvm::SDNode"** %116, align 8
	%117 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	store i32 %115, i32* %117, align 4
	%118 = call i64 @_ZN4llvm12SelectionDAG11getRegisterEjNS_3MVTE(%"struct.llvm::SelectionDAG"* %DAG, i32 19, %"struct.llvm::MVT"* byval align 4 %PtrVT) nounwind		; <i64> [#uses=2]
	%119 = trunc i64 %118 to i32		; <i32> [#uses=1]
	%sroa.store.elt137 = lshr i64 %118, 32		; <i64> [#uses=1]
	%120 = trunc i64 %sroa.store.elt137 to i32		; <i32> [#uses=1]
	%121 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 3, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	%tmp140 = inttoptr i32 %119 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp140, %"struct.llvm::SDNode"** %121, align 8
	%122 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 3, i32 1		; <i32*> [#uses=1]
	store i32 %120, i32* %122, align 4
	%123 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 4, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp110, %"struct.llvm::SDNode"** %123, align 8
	%124 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 4, i32 1		; <i32*> [#uses=1]
	store i32 1, i32* %124, align 4
	%125 = getelementptr [5 x %"struct.llvm::SDValue"]* %Ops1, i32 0, i32 0		; <%"struct.llvm::SDValue"*> [#uses=1]
	%126 = call i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocENS_8SDVTListEPKNS_7SDValueEj(%"struct.llvm::SelectionDAG"* %DAG, i32 195, i32 %7, %"struct.llvm::SDVTList"* byval align 4 %NodeTys, %"struct.llvm::SDValue"* %125, i32 5) nounwind		; <i64> [#uses=2]
	%127 = trunc i64 %126 to i32		; <i32> [#uses=1]
	%sroa.store.elt143 = lshr i64 %126, 32		; <i64> [#uses=1]
	%128 = trunc i64 %sroa.store.elt143 to i32		; <i32> [#uses=1]
	%tmp146 = inttoptr i32 %127 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=3]
	%tmp171195 = getelementptr %"struct.llvm::MVT"* %PtrVT, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp197 = load i32* %tmp171195, align 1		; <i32> [#uses=2]
	%129 = getelementptr %"struct.llvm::MVT"* %VT, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %tmp197, i32* %129, align 8
	%130 = getelementptr %"struct.llvm::MVT"* %VT1.i, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %tmp197, i32* %130, align 8
	%131 = getelementptr %"struct.llvm::MVT"* %VT2.i, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %131, align 8
	%132 = getelementptr %"struct.llvm::MVT"* %VT3.i, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 12, i32* %132, align 8
	%133 = call i64 @_ZN4llvm12SelectionDAG9getVTListENS_3MVTES1_S1_(%"struct.llvm::SelectionDAG"* %DAG, %"struct.llvm::MVT"* byval align 4 %VT1.i, %"struct.llvm::MVT"* byval align 4 %VT2.i, %"struct.llvm::MVT"* byval align 4 %VT3.i) nounwind		; <i64> [#uses=1]
	%134 = trunc i64 %133 to i32		; <i32> [#uses=1]
	%tmp4.i.i = inttoptr i32 %134 to %"struct.llvm::MVT"*		; <%"struct.llvm::MVT"*> [#uses=1]
	%135 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops.i, i32 0, i32 0, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp146, %"struct.llvm::SDNode"** %135, align 8
	%136 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops.i, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %128, i32* %136, align 4
	%137 = call i64 @_ZN4llvm12SelectionDAG11getRegisterEjNS_3MVTE(%"struct.llvm::SelectionDAG"* %DAG, i32 17, %"struct.llvm::MVT"* byval align 4 %VT) nounwind		; <i64> [#uses=2]
	%138 = trunc i64 %137 to i32		; <i32> [#uses=1]
	%sroa.store.elt.i = lshr i64 %137, 32		; <i64> [#uses=1]
	%139 = trunc i64 %sroa.store.elt.i to i32		; <i32> [#uses=1]
	%140 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops.i, i32 0, i32 1, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	%tmp5.i = inttoptr i32 %138 to %"struct.llvm::SDNode"*		; <%"struct.llvm::SDNode"*> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp5.i, %"struct.llvm::SDNode"** %140, align 8
	%141 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops.i, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	store i32 %139, i32* %141, align 4
	%142 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops.i, i32 0, i32 2, i32 0		; <%"struct.llvm::SDNode"**> [#uses=1]
	store %"struct.llvm::SDNode"* %tmp146, %"struct.llvm::SDNode"** %142, align 8
	%143 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops.i, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	store i32 1, i32* %143, align 4
	%144 = icmp eq %"struct.llvm::SDNode"* %tmp146, null		; <i1> [#uses=1]
	%iftmp.588.0.i = select i1 %144, i32 2, i32 3		; <i32> [#uses=1]
	%145 = getelementptr [3 x %"struct.llvm::SDValue"]* %Ops.i, i32 0, i32 0		; <%"struct.llvm::SDValue"*> [#uses=1]
	%146 = call i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocEPKNS_3MVTEjPKNS_7SDValueEj(%"struct.llvm::SelectionDAG"* %DAG, i32 37, i32 %7, %"struct.llvm::MVT"* %tmp4.i.i, i32 3, %"struct.llvm::SDValue"* %145, i32 %iftmp.588.0.i) nounwind		; <i64> [#uses=1]
	ret i64 %146
}

declare void @__assert_rtn(i8*, i8*, i32, i8*) noreturn

declare i64 @_ZN4llvm12SelectionDAG16getGlobalAddressEPKNS_11GlobalValueENS_3MVTExb(%"struct.llvm::SelectionDAG"*, %"struct.llvm::GlobalValue"*, %"struct.llvm::MVT"* byval align 4, i64, i8 zeroext)

declare i64 @_ZN4llvm12SelectionDAG9getVTListENS_3MVTES1_(%"struct.llvm::SelectionDAG"*, %"struct.llvm::MVT"* byval align 4, %"struct.llvm::MVT"* byval align 4)

declare i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocENS_8SDVTListEPKNS_7SDValueEj(%"struct.llvm::SelectionDAG"*, i32, i32, %"struct.llvm::SDVTList"* byval align 4, %"struct.llvm::SDValue"*, i32)

declare i64 @_ZN4llvm12SelectionDAG11getRegisterEjNS_3MVTE(%"struct.llvm::SelectionDAG"*, i32, %"struct.llvm::MVT"* byval align 4)

declare i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocEPKNS_3MVTEjPKNS_7SDValueEj(%"struct.llvm::SelectionDAG"*, i32, i32, %"struct.llvm::MVT"*, i32, %"struct.llvm::SDValue"*, i32)

declare i64 @_ZN4llvm12SelectionDAG9getVTListENS_3MVTES1_S1_(%"struct.llvm::SelectionDAG"*, %"struct.llvm::MVT"* byval align 4, %"struct.llvm::MVT"* byval align 4, %"struct.llvm::MVT"* byval align 4)

declare i64 @_ZN4llvm12SelectionDAG23getTargetExternalSymbolEPKcNS_3MVTE(%"struct.llvm::SelectionDAG"*, i8*, %"struct.llvm::MVT"* byval align 4)

declare i64 @_ZN4llvm12SelectionDAG7getNodeEjNS_8DebugLocENS_3MVTE(%"struct.llvm::SelectionDAG"*, i32, i32, %"struct.llvm::MVT"* byval align 4)

declare void @_ZNK4llvm6SDNode4dumpEv(%"struct.llvm::SDNode"*)
