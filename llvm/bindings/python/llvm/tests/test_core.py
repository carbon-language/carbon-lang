from .base import TestBase
from ..core import OpCode
from ..core import MemoryBuffer
from ..core import PassRegistry
from ..core import Context
from ..core import Module

class TestCore(TestBase):
    def test_opcode(self):
        self.assertTrue(hasattr(OpCode, 'Ret'))
        self.assertTrue(isinstance(OpCode.Ret, OpCode))
        self.assertEqual(OpCode.Ret.value, 1)

        op = OpCode.from_value(1)
        self.assertTrue(isinstance(op, OpCode))
        self.assertEqual(op, OpCode.Ret)

    def test_memory_buffer_create_from_file(self):
        source = self.get_test_file()

        MemoryBuffer(filename=source)

    def test_memory_buffer_failing(self):
        with self.assertRaises(Exception):
            MemoryBuffer(filename="/hopefully/this/path/doesnt/exist")

    def test_memory_buffer_len(self):
        source = self.get_test_file()
        m = MemoryBuffer(filename=source)
        self.assertEqual(len(m), 50)

    def test_create_passregistry(self):
        PassRegistry()

    def test_create_context(self):
        Context.GetGlobalContext()

    def test_create_module_with_name(self):
        # Make sure we can not create a module without a LLVMModuleRef.
        with self.assertRaises(RuntimeError):
            m = Module()
        m = Module.CreateWithName("test-module")

    def test_module_getset_datalayout(self):
        m = Module.CreateWithName("test-module")
        dl = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
        m.datalayout = dl
        self.assertEqual(m.datalayout, dl)

    def test_module_getset_target(self):
        m = Module.CreateWithName("test-module")
        m.target = "thumbv7-apple-ios5.0.0"
        self.assertEqual(m.target, target)

    def test_module_print_module_to_file(self):
        m = Module.CreateWithName("test")
        dl = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
        m.datalayout = dl
        target = "thumbv7-apple-ios5.0.0"
        m.target = target
        m.print_module_to_file("test2.ll")
    
