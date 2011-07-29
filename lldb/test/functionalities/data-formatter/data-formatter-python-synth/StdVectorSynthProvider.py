class StdVectorSynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj;
        self.update()
    def num_children(self):
        start_val = int(self.Mstart.GetValue(),0)
        finish_val = int(self.Mfinish.GetValue(),0)
        return (finish_val-start_val)/self.data_size
    def get_child_index(self,name):
        if name == "len":
            return self.num_children();
        else:
            return int(name.lstrip('[').rstrip(']'))
    def get_child_at_index(self,index):
        if index == self.num_children():
            return self.valobj.CreateValueFromExpression("len",str(self.num_children()))
        else:
            offset = index * self.data_size
            return self.Mstart.CreateChildAtOffset('['+str(index)+']',offset,self.data_type)
    def update(self):
        self.Mimpl = self.valobj.GetChildMemberWithName('_M_impl')
        self.Mstart = self.Mimpl.GetChildMemberWithName('_M_start')
        self.Mfinish = self.Mimpl.GetChildMemberWithName('_M_finish')
        self.data_type = self.Mstart.GetType().GetPointeeType()
        self.data_size = self.data_type.GetByteSize()
