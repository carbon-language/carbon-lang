; RUN: llc < %s -mtriple=i386-apple-darwin10 -relocation-model=pic -disable-fp-elim -color-ss-with-regs | grep {movl\[\[:space:\]\]%eax, %ebx}

	%"struct..0$_67" = type { i32, %"struct.llvm::MachineOperand"**, %"struct.llvm::MachineOperand"* }
	%"struct..1$_69" = type { i32 }
	%"struct.llvm::AbstractTypeUser" = type { i32 (...)** }
	%"struct.llvm::AliasAnalysis" = type opaque
	%"struct.llvm::AnalysisResolver" = type { %"struct.std::vector<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >", %"struct.llvm::PMDataManager"* }
	%"struct.llvm::Annotable" = type { %"struct.llvm::Annotation"* }
	%"struct.llvm::Annotation" = type { i32 (...)**, %"struct..1$_69", %"struct.llvm::Annotation"* }
	%"struct.llvm::Argument" = type { %"struct.llvm::Value", %"struct.llvm::ilist_node<llvm::Argument>", %"struct.llvm::Function"* }
	%"struct.llvm::AttrListPtr" = type { %"struct.llvm::AttributeListImpl"* }
	%"struct.llvm::AttributeListImpl" = type opaque
	%"struct.llvm::BasicBlock" = type { %"struct.llvm::Value", %"struct.llvm::ilist_node<llvm::BasicBlock>", %"struct.llvm::iplist<llvm::Instruction,llvm::ilist_traits<llvm::Instruction> >", %"struct.llvm::Function"* }
	%"struct.llvm::BitVector" = type { i32*, i32, i32 }
	%"struct.llvm::BumpPtrAllocator" = type { i8* }
	%"struct.llvm::CalleeSavedInfo" = type { i32, %"struct.llvm::TargetRegisterClass"*, i32 }
	%"struct.llvm::Constant" = type { %"struct.llvm::User" }
	%"struct.llvm::DebugLocTracker" = type { %"struct.std::vector<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >", %"struct.llvm::DenseMap<llvm::DebugLocTuple,unsigned int,llvm::DenseMapInfo<llvm::DebugLocTuple>,llvm::DenseMapInfo<unsigned int> >" }
	%"struct.llvm::DebugLocTuple" = type { %"struct.llvm::GlobalVariable"*, i32, i32 }
	%"struct.llvm::DenseMap<llvm::DebugLocTuple,unsigned int,llvm::DenseMapInfo<llvm::DebugLocTuple>,llvm::DenseMapInfo<unsigned int> >" = type { i32, %"struct.std::pair<llvm::DebugLocTuple,unsigned int>"*, i32, i32 }
	%"struct.llvm::DenseMap<llvm::MachineInstr*,unsigned int,llvm::DenseMapInfo<llvm::MachineInstr*>,llvm::DenseMapInfo<unsigned int> >" = type { i32, %"struct.std::pair<llvm::MachineInstr*,unsigned int>"*, i32, i32 }
	%"struct.llvm::DenseMap<unsigned int,char,llvm::DenseMapInfo<unsigned int>,llvm::DenseMapInfo<char> >" = type { i32, %"struct.std::pair<unsigned int,char>"*, i32, i32 }
	%"struct.llvm::DenseMap<unsigned int,llvm::LiveInterval*,llvm::DenseMapInfo<unsigned int>,llvm::DenseMapInfo<llvm::LiveInterval*> >" = type { i32, %"struct.std::pair<unsigned int,llvm::LiveInterval*>"*, i32, i32 }
	%"struct.llvm::DenseSet<unsigned int,llvm::DenseMapInfo<unsigned int> >" = type { %"struct.llvm::DenseMap<unsigned int,char,llvm::DenseMapInfo<unsigned int>,llvm::DenseMapInfo<char> >" }
	%"struct.llvm::Function" = type { %"struct.llvm::GlobalValue", %"struct.llvm::Annotable", %"struct.llvm::ilist_node<llvm::Function>", %"struct.llvm::iplist<llvm::BasicBlock,llvm::ilist_traits<llvm::BasicBlock> >", %"struct.llvm::iplist<llvm::Argument,llvm::ilist_traits<llvm::Argument> >", %"struct.llvm::ValueSymbolTable"*, %"struct.llvm::AttrListPtr" }
	%"struct.llvm::FunctionPass" = type { %"struct.llvm::Pass" }
	%"struct.llvm::GlobalValue" = type { %"struct.llvm::Constant", %"struct.llvm::Module"*, i32, %"struct.std::string" }
	%"struct.llvm::GlobalVariable" = type opaque
	%"struct.llvm::Instruction" = type { %"struct.llvm::User", %"struct.llvm::ilist_node<llvm::Instruction>", %"struct.llvm::BasicBlock"* }
	%"struct.llvm::LiveInterval" = type <{ i32, float, i16, [6 x i8], %"struct.llvm::SmallVector<llvm::LiveRange,4u>", %"struct.llvm::SmallVector<llvm::MachineBasicBlock*,4u>" }>
	%"struct.llvm::LiveIntervals" = type { %"struct.llvm::MachineFunctionPass", %"struct.llvm::MachineFunction"*, %"struct.llvm::MachineRegisterInfo"*, %"struct.llvm::TargetMachine"*, %"struct.llvm::TargetRegisterInfo"*, %"struct.llvm::TargetInstrInfo"*, %"struct.llvm::AliasAnalysis"*, %"struct.llvm::LiveVariables"*, %"struct.llvm::BumpPtrAllocator", %"struct.std::vector<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >", %"struct.std::vector<std::pair<unsigned int, llvm::MachineBasicBlock*>,std::allocator<std::pair<unsigned int, llvm::MachineBasicBlock*> > >", i64, %"struct.llvm::DenseMap<llvm::MachineInstr*,unsigned int,llvm::DenseMapInfo<llvm::MachineInstr*>,llvm::DenseMapInfo<unsigned int> >", %"struct.std::vector<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >", %"struct.llvm::DenseMap<unsigned int,llvm::LiveInterval*,llvm::DenseMapInfo<unsigned int>,llvm::DenseMapInfo<llvm::LiveInterval*> >", %"struct.llvm::BitVector", %"struct.std::vector<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >" }
	%"struct.llvm::LiveVariables" = type opaque
	%"struct.llvm::MVT" = type { %"struct..1$_69" }
	%"struct.llvm::MachineBasicBlock" = type { %"struct.llvm::ilist_node<llvm::MachineBasicBlock>", %"struct.llvm::ilist<llvm::MachineInstr>", %"struct.llvm::BasicBlock"*, i32, %"struct.llvm::MachineFunction"*, %"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >", %"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >", %"struct.std::vector<int,std::allocator<int> >", i32, i8 }
	%"struct.llvm::MachineConstantPool" = type opaque
	%"struct.llvm::MachineFrameInfo" = type { %"struct.std::vector<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >", i32, i8, i8, i64, i32, i32, i8, i32, i32, %"struct.std::vector<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >", %"struct.llvm::MachineModuleInfo"*, %"struct.llvm::TargetFrameInfo"* }
	%"struct.llvm::MachineFrameInfo::StackObject" = type { i64, i32, i8, i64 }
	%"struct.llvm::MachineFunction" = type { %"struct.llvm::Annotation", %"struct.llvm::Function"*, %"struct.llvm::TargetMachine"*, %"struct.llvm::MachineRegisterInfo"*, %"struct.llvm::AbstractTypeUser"*, %"struct.llvm::MachineFrameInfo"*, %"struct.llvm::MachineConstantPool"*, %"struct.llvm::MachineJumpTableInfo"*, %"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >", %"struct.llvm::BumpPtrAllocator", %"struct.llvm::Recycler<llvm::MachineBasicBlock,80ul,4ul>", %"struct.llvm::Recycler<llvm::MachineBasicBlock,80ul,4ul>", %"struct.llvm::ilist<llvm::MachineBasicBlock>", %"struct..1$_69", %"struct.llvm::DebugLocTracker" }
	%"struct.llvm::MachineFunctionPass" = type { %"struct.llvm::FunctionPass" }
	%"struct.llvm::MachineInstr" = type { %"struct.llvm::ilist_node<llvm::MachineInstr>", %"struct.llvm::TargetInstrDesc"*, i16, %"struct.std::vector<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >", %"struct.std::list<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >", %"struct.llvm::MachineBasicBlock"*, %"struct..1$_69" }
	%"struct.llvm::MachineJumpTableInfo" = type opaque
	%"struct.llvm::MachineModuleInfo" = type opaque
	%"struct.llvm::MachineOperand" = type { i8, i8, i8, %"struct.llvm::MachineInstr"*, %"struct.llvm::MachineOperand::$_66" }
	%"struct.llvm::MachineOperand::$_66" = type { %"struct..0$_67" }
	%"struct.llvm::MachineRegisterInfo" = type { %"struct.std::vector<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >", %"struct.std::vector<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >", %"struct.llvm::MachineOperand"**, %"struct.llvm::BitVector", %"struct.std::vector<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >", %"struct.std::vector<int,std::allocator<int> >" }
	%"struct.llvm::Module" = type opaque
	%"struct.llvm::PATypeHandle" = type { %"struct.llvm::Type"*, %"struct.llvm::AbstractTypeUser"* }
	%"struct.llvm::PATypeHolder" = type { %"struct.llvm::Type"* }
	%"struct.llvm::PMDataManager" = type opaque
	%"struct.llvm::Pass" = type { i32 (...)**, %"struct.llvm::AnalysisResolver"*, i32 }
	%"struct.llvm::PassInfo" = type { i8*, i8*, i32, i8, i8, i8, %"struct.std::vector<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >", %"struct.llvm::Pass"* ()* }
	%"struct.llvm::Recycler<llvm::MachineBasicBlock,80ul,4ul>" = type { %"struct.llvm::iplist<llvm::RecyclerStruct,llvm::ilist_traits<llvm::RecyclerStruct> >" }
	%"struct.llvm::RecyclerStruct" = type { %"struct.llvm::RecyclerStruct"*, %"struct.llvm::RecyclerStruct"* }
	%"struct.llvm::SmallVector<llvm::LiveRange,4u>" = type <{ [17 x i8], [47 x i8] }>
	%"struct.llvm::SmallVector<llvm::MachineBasicBlock*,4u>" = type <{ [17 x i8], [15 x i8] }>
	%"struct.llvm::TargetAsmInfo" = type opaque
	%"struct.llvm::TargetFrameInfo" = type opaque
	%"struct.llvm::TargetInstrDesc" = type { i16, i16, i16, i16, i8*, i32, i32, i32*, i32*, %"struct.llvm::TargetRegisterClass"**, %"struct.llvm::TargetOperandInfo"* }
	%"struct.llvm::TargetInstrInfo" = type { i32 (...)**, %"struct.llvm::TargetInstrDesc"*, i32 }
	%"struct.llvm::TargetMachine" = type { i32 (...)**, %"struct.llvm::TargetAsmInfo"* }
	%"struct.llvm::TargetOperandInfo" = type { i16, i16, i32 }
	%"struct.llvm::TargetRegisterClass" = type { i32 (...)**, i32, i8*, %"struct.llvm::MVT"*, %"struct.llvm::TargetRegisterClass"**, %"struct.llvm::TargetRegisterClass"**, %"struct.llvm::TargetRegisterClass"**, %"struct.llvm::TargetRegisterClass"**, i32, i32, i32, i32*, i32*, %"struct.llvm::DenseSet<unsigned int,llvm::DenseMapInfo<unsigned int> >" }
	%"struct.llvm::TargetRegisterDesc" = type { i8*, i8*, i32*, i32*, i32* }
	%"struct.llvm::TargetRegisterInfo" = type { i32 (...)**, i32*, i32, i32*, i32, i32*, i32, %"struct.llvm::TargetRegisterDesc"*, i32, %"struct.llvm::TargetRegisterClass"**, %"struct.llvm::TargetRegisterClass"**, i32, i32 }
	%"struct.llvm::Type" = type { %"struct.llvm::AbstractTypeUser", i8, [3 x i8], i32, %"struct.llvm::Type"*, %"struct.std::vector<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >", i32, %"struct.llvm::PATypeHandle"* }
	%"struct.llvm::Use" = type { %"struct.llvm::Value"*, %"struct.llvm::Use"*, %"struct..1$_69" }
	%"struct.llvm::User" = type { %"struct.llvm::Value", %"struct.llvm::Use"*, i32 }
	%"struct.llvm::Value" = type { i32 (...)**, i8, i8, i16, %"struct.llvm::PATypeHolder", %"struct.llvm::Use"*, %"struct.llvm::ValueName"* }
	%"struct.llvm::ValueName" = type opaque
	%"struct.llvm::ValueSymbolTable" = type opaque
	%"struct.llvm::ilist<llvm::MachineBasicBlock>" = type { %"struct.llvm::iplist<llvm::MachineBasicBlock,llvm::ilist_traits<llvm::MachineBasicBlock> >" }
	%"struct.llvm::ilist<llvm::MachineInstr>" = type { %"struct.llvm::iplist<llvm::MachineInstr,llvm::ilist_traits<llvm::MachineInstr> >" }
	%"struct.llvm::ilist_node<llvm::Argument>" = type { %"struct.llvm::Argument"*, %"struct.llvm::Argument"* }
	%"struct.llvm::ilist_node<llvm::BasicBlock>" = type { %"struct.llvm::BasicBlock"*, %"struct.llvm::BasicBlock"* }
	%"struct.llvm::ilist_node<llvm::Function>" = type { %"struct.llvm::Function"*, %"struct.llvm::Function"* }
	%"struct.llvm::ilist_node<llvm::Instruction>" = type { %"struct.llvm::Instruction"*, %"struct.llvm::Instruction"* }
	%"struct.llvm::ilist_node<llvm::MachineBasicBlock>" = type { %"struct.llvm::MachineBasicBlock"*, %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::ilist_node<llvm::MachineInstr>" = type { %"struct.llvm::MachineInstr"*, %"struct.llvm::MachineInstr"* }
	%"struct.llvm::ilist_traits<llvm::Argument>" = type { %"struct.llvm::ilist_node<llvm::Argument>" }
	%"struct.llvm::ilist_traits<llvm::BasicBlock>" = type { %"struct.llvm::ilist_node<llvm::BasicBlock>" }
	%"struct.llvm::ilist_traits<llvm::Instruction>" = type { %"struct.llvm::ilist_node<llvm::Instruction>" }
	%"struct.llvm::ilist_traits<llvm::MachineBasicBlock>" = type { %"struct.llvm::ilist_node<llvm::MachineBasicBlock>" }
	%"struct.llvm::ilist_traits<llvm::MachineInstr>" = type { %"struct.llvm::ilist_node<llvm::MachineInstr>", %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::ilist_traits<llvm::RecyclerStruct>" = type { %"struct.llvm::RecyclerStruct" }
	%"struct.llvm::iplist<llvm::Argument,llvm::ilist_traits<llvm::Argument> >" = type { %"struct.llvm::ilist_traits<llvm::Argument>", %"struct.llvm::Argument"* }
	%"struct.llvm::iplist<llvm::BasicBlock,llvm::ilist_traits<llvm::BasicBlock> >" = type { %"struct.llvm::ilist_traits<llvm::BasicBlock>", %"struct.llvm::BasicBlock"* }
	%"struct.llvm::iplist<llvm::Instruction,llvm::ilist_traits<llvm::Instruction> >" = type { %"struct.llvm::ilist_traits<llvm::Instruction>", %"struct.llvm::Instruction"* }
	%"struct.llvm::iplist<llvm::MachineBasicBlock,llvm::ilist_traits<llvm::MachineBasicBlock> >" = type { %"struct.llvm::ilist_traits<llvm::MachineBasicBlock>", %"struct.llvm::MachineBasicBlock"* }
	%"struct.llvm::iplist<llvm::MachineInstr,llvm::ilist_traits<llvm::MachineInstr> >" = type { %"struct.llvm::ilist_traits<llvm::MachineInstr>", %"struct.llvm::MachineInstr"* }
	%"struct.llvm::iplist<llvm::RecyclerStruct,llvm::ilist_traits<llvm::RecyclerStruct> >" = type { %"struct.llvm::ilist_traits<llvm::RecyclerStruct>", %"struct.llvm::RecyclerStruct"* }
	%"struct.std::IdxMBBPair" = type { i32, %"struct.llvm::MachineBasicBlock"* }
	%"struct.std::_List_base<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >" = type { %"struct.llvm::ilist_traits<llvm::RecyclerStruct>" }
	%"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" = type { %"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >::_Vector_impl" }
	%"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >::_Vector_impl" = type { %"struct.llvm::PassInfo"**, %"struct.llvm::PassInfo"**, %"struct.llvm::PassInfo"** }
	%"struct.std::_Vector_base<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" }
	%"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" = type { i32*, i32*, i32* }
	%"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" = type { %"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >::_Vector_impl" = type { %"struct.llvm::AbstractTypeUser"**, %"struct.llvm::AbstractTypeUser"**, %"struct.llvm::AbstractTypeUser"** }
	%"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >" = type { %"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >::_Vector_impl" = type { %"struct.llvm::CalleeSavedInfo"*, %"struct.llvm::CalleeSavedInfo"*, %"struct.llvm::CalleeSavedInfo"* }
	%"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >" = type { %"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >::_Vector_impl" = type { %"struct.llvm::DebugLocTuple"*, %"struct.llvm::DebugLocTuple"*, %"struct.llvm::DebugLocTuple"* }
	%"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >" = type { %"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >::_Vector_impl" = type { %"struct.llvm::MachineBasicBlock"**, %"struct.llvm::MachineBasicBlock"**, %"struct.llvm::MachineBasicBlock"** }
	%"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >" = type { %"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >::_Vector_impl" = type { %"struct.llvm::MachineFrameInfo::StackObject"*, %"struct.llvm::MachineFrameInfo::StackObject"*, %"struct.llvm::MachineFrameInfo::StackObject"* }
	%"struct.std::_Vector_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >" = type { %"struct.std::_Vector_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >::_Vector_impl" = type { %"struct.llvm::MachineInstr"**, %"struct.llvm::MachineInstr"**, %"struct.llvm::MachineInstr"** }
	%"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" = type { %"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >::_Vector_impl" }
	%"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >::_Vector_impl" = type { %"struct.llvm::MachineOperand"*, %"struct.llvm::MachineOperand"*, %"struct.llvm::MachineOperand"* }
	%"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >::_Vector_impl" = type { %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"*, %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"*, %"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>"* }
	%"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >::_Vector_impl" = type { %"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>"*, %"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>"*, %"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>"* }
	%"struct.std::_Vector_base<std::pair<unsigned int, llvm::MachineBasicBlock*>,std::allocator<std::pair<unsigned int, llvm::MachineBasicBlock*> > >" = type { %"struct.std::_Vector_base<std::pair<unsigned int, llvm::MachineBasicBlock*>,std::allocator<std::pair<unsigned int, llvm::MachineBasicBlock*> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<unsigned int, llvm::MachineBasicBlock*>,std::allocator<std::pair<unsigned int, llvm::MachineBasicBlock*> > >::_Vector_impl" = type { %"struct.std::IdxMBBPair"*, %"struct.std::IdxMBBPair"*, %"struct.std::IdxMBBPair"* }
	%"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >" = type { %"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >::_Vector_impl" = type { %"struct.std::pair<unsigned int,int>"*, %"struct.std::pair<unsigned int,int>"*, %"struct.std::pair<unsigned int,int>"* }
	%"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >" = type { %"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >::_Vector_impl" = type { %"struct.std::vector<int,std::allocator<int> >"*, %"struct.std::vector<int,std::allocator<int> >"*, %"struct.std::vector<int,std::allocator<int> >"* }
	%"struct.std::list<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >" = type { %"struct.std::_List_base<llvm::MachineMemOperand,std::allocator<llvm::MachineMemOperand> >" }
	%"struct.std::pair<const llvm::PassInfo*,llvm::Pass*>" = type { %"struct.llvm::PassInfo"*, %"struct.llvm::Pass"* }
	%"struct.std::pair<const llvm::TargetRegisterClass*,llvm::MachineOperand*>" = type { %"struct.llvm::TargetRegisterClass"*, %"struct.llvm::MachineOperand"* }
	%"struct.std::pair<llvm::DebugLocTuple,unsigned int>" = type { %"struct.llvm::DebugLocTuple", i32 }
	%"struct.std::pair<llvm::MachineInstr*,unsigned int>" = type { %"struct.llvm::MachineInstr"*, i32 }
	%"struct.std::pair<unsigned int,char>" = type { i32, i8 }
	%"struct.std::pair<unsigned int,int>" = type { i32, i32 }
	%"struct.std::pair<unsigned int,llvm::LiveInterval*>" = type { i32, %"struct.llvm::LiveInterval"* }
	%"struct.std::string" = type { %"struct.llvm::BumpPtrAllocator" }
	%"struct.std::vector<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" = type { %"struct.std::_Vector_base<const llvm::PassInfo*,std::allocator<const llvm::PassInfo*> >" }
	%"struct.std::vector<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >" }
	%"struct.std::vector<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" = type { %"struct.std::_Vector_base<llvm::AbstractTypeUser*,std::allocator<llvm::AbstractTypeUser*> >" }
	%"struct.std::vector<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >" = type { %"struct.std::_Vector_base<llvm::CalleeSavedInfo,std::allocator<llvm::CalleeSavedInfo> >" }
	%"struct.std::vector<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >" = type { %"struct.std::_Vector_base<llvm::DebugLocTuple,std::allocator<llvm::DebugLocTuple> >" }
	%"struct.std::vector<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >" = type { %"struct.std::_Vector_base<llvm::MachineBasicBlock*,std::allocator<llvm::MachineBasicBlock*> >" }
	%"struct.std::vector<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >" = type { %"struct.std::_Vector_base<llvm::MachineFrameInfo::StackObject,std::allocator<llvm::MachineFrameInfo::StackObject> >" }
	%"struct.std::vector<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >" = type { %"struct.std::_Vector_base<llvm::MachineInstr*,std::allocator<llvm::MachineInstr*> >" }
	%"struct.std::vector<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" = type { %"struct.std::_Vector_base<llvm::MachineOperand,std::allocator<llvm::MachineOperand> >" }
	%"struct.std::vector<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::PassInfo*, llvm::Pass*>,std::allocator<std::pair<const llvm::PassInfo*, llvm::Pass*> > >" }
	%"struct.std::vector<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >" = type { %"struct.std::_Vector_base<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*>,std::allocator<std::pair<const llvm::TargetRegisterClass*, llvm::MachineOperand*> > >" }
	%"struct.std::vector<std::pair<unsigned int, llvm::MachineBasicBlock*>,std::allocator<std::pair<unsigned int, llvm::MachineBasicBlock*> > >" = type { %"struct.std::_Vector_base<std::pair<unsigned int, llvm::MachineBasicBlock*>,std::allocator<std::pair<unsigned int, llvm::MachineBasicBlock*> > >" }
	%"struct.std::vector<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >" = type { %"struct.std::_Vector_base<std::pair<unsigned int, unsigned int>,std::allocator<std::pair<unsigned int, unsigned int> > >" }
	%"struct.std::vector<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >" = type { %"struct.std::_Vector_base<std::vector<unsigned int, std::allocator<unsigned int> >,std::allocator<std::vector<unsigned int, std::allocator<unsigned int> > > >" }
@_ZZNK4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE15LookupBucketForERKS2_RPSt4pairIS2_jEE8__func__ = external constant [16 x i8]		; <[16 x i8]*> [#uses=1]
@"\01LC6" = external constant [56 x i8]		; <[56 x i8]*> [#uses=1]
@"\01LC7" = external constant [134 x i8]		; <[134 x i8]*> [#uses=1]
@"\01LC8" = external constant [72 x i8]		; <[72 x i8]*> [#uses=1]
@_ZZN4llvm13LiveIntervals24InsertMachineInstrInMapsEPNS_12MachineInstrEjE8__func__ = external constant [25 x i8]		; <[25 x i8]*> [#uses=1]
@"\01LC51" = external constant [42 x i8]		; <[42 x i8]*> [#uses=1]

define void @_ZN4llvm13LiveIntervals24InsertMachineInstrInMapsEPNS_12MachineInstrEj(%"struct.llvm::LiveIntervals"* nocapture %this, %"struct.llvm::MachineInstr"* %MI, i32 %Index) nounwind ssp {
entry:
	%0 = call i64 @_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE4findERKS2_(%"struct.llvm::DenseMap<llvm::MachineInstr*,unsigned int,llvm::DenseMapInfo<llvm::MachineInstr*>,llvm::DenseMapInfo<unsigned int> >"* null, %"struct.llvm::MachineInstr"** null) nounwind ssp		; <i64> [#uses=1]
	%1 = trunc i64 %0 to i32		; <i32> [#uses=1]
	%tmp11 = inttoptr i32 %1 to %"struct.std::pair<llvm::MachineInstr*,unsigned int>"*		; <%"struct.std::pair<llvm::MachineInstr*,unsigned int>"*> [#uses=1]
	%2 = load %"struct.std::pair<llvm::MachineInstr*,unsigned int>"** null, align 4		; <%"struct.std::pair<llvm::MachineInstr*,unsigned int>"*> [#uses=3]
	%3 = getelementptr %"struct.llvm::LiveIntervals"* %this, i32 0, i32 12, i32 0		; <i32*> [#uses=1]
	%4 = load i32* %3, align 4		; <i32> [#uses=2]
	%5 = getelementptr %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %2, i32 %4		; <%"struct.std::pair<llvm::MachineInstr*,unsigned int>"*> [#uses=1]
	br label %bb1.i.i.i

bb.i.i.i:		; preds = %bb2.i.i.i
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb1.i.i.i

bb1.i.i.i:		; preds = %bb.i.i.i, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb.i.i.i ]		; <i32> [#uses=2]
	%tmp32 = shl i32 %indvar, 3		; <i32> [#uses=1]
	%ctg2.sum = add i32 0, %tmp32		; <i32> [#uses=1]
	%ctg237 = getelementptr i8* null, i32 %ctg2.sum		; <i8*> [#uses=1]
	%.0.0.i = bitcast i8* %ctg237 to %"struct.std::pair<llvm::MachineInstr*,unsigned int>"*		; <%"struct.std::pair<llvm::MachineInstr*,unsigned int>"*> [#uses=2]
	%6 = icmp eq %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %.0.0.i, %5		; <i1> [#uses=1]
	br i1 %6, label %_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE3endEv.exit, label %bb2.i.i.i

bb2.i.i.i:		; preds = %bb1.i.i.i
	%7 = load %"struct.llvm::MachineInstr"** null, align 4		; <%"struct.llvm::MachineInstr"*> [#uses=1]
	%8 = icmp eq %"struct.llvm::MachineInstr"* %7, inttoptr (i32 -8 to %"struct.llvm::MachineInstr"*)		; <i1> [#uses=1]
	%or.cond.i.i.i21 = or i1 false, %8		; <i1> [#uses=1]
	br i1 %or.cond.i.i.i21, label %bb.i.i.i, label %_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE3endEv.exit

_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE3endEv.exit:		; preds = %bb2.i.i.i, %bb1.i.i.i
	%9 = icmp eq %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %tmp11, %.0.0.i		; <i1> [#uses=1]
	br i1 %9, label %bb7, label %bb6

bb6:		; preds = %_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE3endEv.exit
	call void @__assert_rtn(i8* getelementptr ([25 x i8]* @_ZZN4llvm13LiveIntervals24InsertMachineInstrInMapsEPNS_12MachineInstrEjE8__func__, i32 0, i32 0), i8* getelementptr ([72 x i8]* @"\01LC8", i32 0, i32 0), i32 251, i8* getelementptr ([42 x i8]* @"\01LC51", i32 0, i32 0)) noreturn nounwind
	unreachable

bb7:		; preds = %_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE3endEv.exit
	%10 = load %"struct.llvm::MachineInstr"** null, align 4		; <%"struct.llvm::MachineInstr"*> [#uses=2]
	%11 = icmp eq %"struct.llvm::MachineInstr"* %10, inttoptr (i32 -8 to %"struct.llvm::MachineInstr"*)		; <i1> [#uses=1]
	%or.cond40.i.i.i = or i1 false, %11		; <i1> [#uses=1]
	br i1 %or.cond40.i.i.i, label %bb5.i.i.i, label %bb6.preheader.i.i.i

bb6.preheader.i.i.i:		; preds = %bb7
	%12 = add i32 %4, -1		; <i32> [#uses=1]
	br label %bb6.i.i.i

bb5.i.i.i:		; preds = %bb7
	call void @__assert_rtn(i8* getelementptr ([16 x i8]* @_ZZNK4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE15LookupBucketForERKS2_RPSt4pairIS2_jEE8__func__, i32 0, i32 0), i8* getelementptr ([56 x i8]* @"\01LC6", i32 0, i32 0), i32 390, i8* getelementptr ([134 x i8]* @"\01LC7", i32 0, i32 0)) noreturn nounwind
	unreachable

bb6.i.i.i:		; preds = %bb17.i.i.i, %bb6.preheader.i.i.i
	%FoundTombstone.1.i.i.i = phi %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* [ %FoundTombstone.0.i.i.i, %bb17.i.i.i ], [ null, %bb6.preheader.i.i.i ]		; <%"struct.std::pair<llvm::MachineInstr*,unsigned int>"*> [#uses=2]
	%ProbeAmt.0.i.i.i = phi i32 [ 0, %bb17.i.i.i ], [ 1, %bb6.preheader.i.i.i ]		; <i32> [#uses=1]
	%BucketNo.0.i.i.i = phi i32 [ %20, %bb17.i.i.i ], [ 0, %bb6.preheader.i.i.i ]		; <i32> [#uses=2]
	%13 = and i32 %BucketNo.0.i.i.i, %12		; <i32> [#uses=2]
	%14 = getelementptr %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %2, i32 %13		; <%"struct.std::pair<llvm::MachineInstr*,unsigned int>"*> [#uses=2]
	%15 = getelementptr %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %2, i32 %13, i32 0		; <%"struct.llvm::MachineInstr"**> [#uses=1]
	%16 = load %"struct.llvm::MachineInstr"** %15, align 4		; <%"struct.llvm::MachineInstr"*> [#uses=2]
	%17 = icmp eq %"struct.llvm::MachineInstr"* %16, %10		; <i1> [#uses=1]
	br i1 %17, label %_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEEixERKS2_.exit, label %bb17.i.i.i

bb17.i.i.i:		; preds = %bb6.i.i.i
	%18 = icmp eq %"struct.llvm::MachineInstr"* %16, inttoptr (i32 -8 to %"struct.llvm::MachineInstr"*)		; <i1> [#uses=1]
	%19 = icmp eq %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %FoundTombstone.1.i.i.i, null		; <i1> [#uses=1]
	%or.cond.i.i.i = and i1 %18, %19		; <i1> [#uses=1]
	%FoundTombstone.0.i.i.i = select i1 %or.cond.i.i.i, %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %14, %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %FoundTombstone.1.i.i.i		; <%"struct.std::pair<llvm::MachineInstr*,unsigned int>"*> [#uses=1]
	%20 = add i32 %BucketNo.0.i.i.i, %ProbeAmt.0.i.i.i		; <i32> [#uses=1]
	br label %bb6.i.i.i

_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEEixERKS2_.exit:		; preds = %bb6.i.i.i
	%21 = getelementptr %"struct.std::pair<llvm::MachineInstr*,unsigned int>"* %14, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %Index, i32* %21, align 4
	ret void
}

declare void @__assert_rtn(i8*, i8*, i32, i8*) noreturn

declare i64 @_ZN4llvm8DenseMapIPNS_12MachineInstrEjNS_12DenseMapInfoIS2_EENS3_IjEEE4findERKS2_(%"struct.llvm::DenseMap<llvm::MachineInstr*,unsigned int,llvm::DenseMapInfo<llvm::MachineInstr*>,llvm::DenseMapInfo<unsigned int> >"* nocapture, %"struct.llvm::MachineInstr"** nocapture) nounwind ssp
