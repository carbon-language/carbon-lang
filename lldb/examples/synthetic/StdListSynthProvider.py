import re
class StdListSynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj;
        self.update()
    def num_children(self):
        next_val = int(self.Mnext.GetValue(),0)
        prev_val = int(self.Mprev.GetValue(),0)
        if next_val == 0:
        	return 0;
        if next_val == self.Mnode_address:
        	return 0;
        if next_val == prev_val:
        	return 1;
        size = 2
        current = self.Mnext
        while int(current.GetChildMemberWithName('_M_next').GetValue(),0) != self.Mnode_address:
        	size = size + 1;
        	current = current.GetChildMemberWithName('_M_next')
        return (size - 1)
    def get_child_index(self,name):
        if name == "len":
            return self.num_children();
        else:
            return int(name.lstrip('[').rstrip(']'))
    def get_child_at_index(self,index):
        if index == self.num_children():
            return self.valobj.CreateValueFromExpression("len",str(self.num_children()))
        else:
            offset = index
            current = self.Mnext;
            while offset > 0:
            	current = current.GetChildMemberWithName('_M_next');
            	offset = offset - 1;
            return current.CreateChildAtOffset('['+str(index)+']',2*current.GetType().GetByteSize(),self.data_type)
    def extract_type_name(self,name):
        self.type_name = name[16:]
        index = 2
        count_of_template = 1
        while index < len(self.type_name):
            if self.type_name[index] == '<':
                count_of_template = count_of_template + 1;
            elif self.type_name[index] == '>':
                count_of_template = count_of_template - 1;
            elif self.type_name[index] == ',' and count_of_template == 1:
                self.type_name = self.type_name[:index]
                break
            index = index + 1;
        self.type_name_nospaces = self.type_name.replace(", ", ",")
    def update(self):
        self.Mimpl = self.valobj.GetChildMemberWithName('_M_impl')
        self.Mnode = self.Mimpl.GetChildMemberWithName('_M_node')
        self.extract_type_name(self.Mimpl.GetType().GetName())
        self.Mnode_address = int(self.valobj.AddressOf().GetValue(), 0)
        self.Mnext = self.Mnode.GetChildMemberWithName('_M_next')
        self.Mprev = self.Mnode.GetChildMemberWithName('_M_prev')
        self.data_type = self.Mnode.GetTarget().FindFirstType(self.type_name)
        # tries to fight against a difference in formatting type names between gcc and clang
        if self.data_type.IsValid() == False:
            self.data_type = self.Mnode.GetTarget().FindFirstType(self.type_name_nospaces)
        self.data_size = self.data_type.GetByteSize()
