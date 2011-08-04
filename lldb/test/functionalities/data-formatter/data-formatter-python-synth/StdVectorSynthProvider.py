class StdVectorSynthProvider:

	def __init__(self, valobj, dict):
		self.valobj = valobj;
		self.update()

	def num_children(self):
		start_val = self.start.GetValueAsUnsigned(0)
		finish_val = self.finish.GetValueAsUnsigned(0)
		end_val  = self.end.GetValueAsUnsigned(0)
		# Before a vector has been constructed, it will contain bad values
		# so we really need to be careful about the length we return since
		# unitialized data can cause us to return a huge number. We need
		# to also check for any of the start, finish or end of storage values
		# being zero (NULL). If any are, then this vector has not been 
		# initialized yet and we should return zero
        
		# Make sure nothing is NULL
		if start_val == 0 or finish_val == 0 or end_val == 0:
			return 0
		# Make sure start is less than finish
		if start_val >= finish_val:
			return 0
		# Make sure finish is less than or equal to end of storage
		if finish_val > end_val:
			return 0

		# We might still get things wrong, so cap things at 256 items for now
		# TODO: read a target "settings set" variable for this to allow it to
		# be customized
		num_children = (finish_val-start_val)/self.data_size
		if num_children > 256:
			return 256
		return num_children

	def get_child_index(self,name):
		return int(name.lstrip('[').rstrip(']'))

	def get_child_at_index(self,index):
		if index >= self.num_children():
			return None;
		offset = index * self.data_size
		return self.start.CreateChildAtOffset('['+str(index)+']',offset,self.data_type)

	def update(self):
		impl = self.valobj.GetChildMemberWithName('_M_impl')
		self.start = impl.GetChildMemberWithName('_M_start')
		self.finish = impl.GetChildMemberWithName('_M_finish')
		self.end = impl.GetChildMemberWithName('_M_end_of_storage')
		self.data_type = self.start.GetType().GetPointeeType()
		self.data_size = self.data_type.GetByteSize()

