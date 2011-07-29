import lldb
class ftsp:
	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()
	def num_children(self):
		if self.char.IsValid():
			return 4;
		return 0;
	def get_child_index(self,name):
		return 0;
	def get_child_at_index(self,index):
		if index == 0:
			return self.x.Cast(self.char)
		return self.x.CreateChildAtOffset(str(index),
									   index,
									   self.char);
	def update(self):
		self.x = self.valobj.GetChildMemberWithName('x');
		self.char = self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar)
