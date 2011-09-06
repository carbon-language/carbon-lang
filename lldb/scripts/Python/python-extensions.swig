
%extend lldb::SBAddress {
        PyObject *lldb::SBAddress::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBBlock {
        PyObject *lldb::SBBlock::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBBreakpoint {
        PyObject *lldb::SBBreakpoint::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBBreakpointLocation {
        PyObject *lldb::SBBreakpointLocation::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelFull);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBCommandReturnObject {
        PyObject *lldb::SBCommandReturnObject::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBCompileUnit {
        PyObject *lldb::SBCompileUnit::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBData {
        PyObject *lldb::SBData::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBDebugger {
        PyObject *lldb::SBDebugger::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBError {
        PyObject *lldb::SBError::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBFileSpec {
        PyObject *lldb::SBFileSpec::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBFrame {
        PyObject *lldb::SBFrame::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBFunction {
        PyObject *lldb::SBFunction::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBInstruction {
        PyObject *lldb::SBInstruction::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBInstructionList {
        PyObject *lldb::SBInstructionList::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBLineEntry {
        PyObject *lldb::SBLineEntry::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBModule {
        PyObject *lldb::SBModule::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBProcess {
        PyObject *lldb::SBProcess::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBSymbol {
        PyObject *lldb::SBSymbol::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBSymbolContext {
        PyObject *lldb::SBSymbolContext::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBTarget {
        PyObject *lldb::SBTarget::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description, lldb::eDescriptionLevelBrief);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBThread {
        PyObject *lldb::SBThread::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}
%extend lldb::SBValue {
        PyObject *lldb::SBValue::__repr__ (){
                lldb::SBStream description;
                $self->GetDescription (description);
                return PyString_FromString (description.GetData());
        }
}

