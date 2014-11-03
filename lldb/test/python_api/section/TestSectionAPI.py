"""
Test SBSection APIs.
"""

import unittest2
from lldbtest import *

class SectionAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_get_target_byte_size_with_dsym(self):
        d = {'EXE': 'a.out'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.create_simple_target('a.out')

        # find the .data section of the main module            
        data_section = self.find_data_section(target)

        self.assertEquals(data_section.target_byte_size, 1)

    @python_api_test
    @dwarf_test
    def test_get_target_byte_size_with_dwarf(self):
        d = {'EXE': 'b.out'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.create_simple_target('b.out')

        # find the .data section of the main module            
        data_section = self.find_data_section(target)

        self.assertEquals(data_section.target_byte_size, 1)

    def create_simple_target(self, fn):
        exe = os.path.join(os.getcwd(), fn)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        return target

    def find_data_section(self, target):
        mod = target.GetModuleAtIndex(0)
        data_section = None
        for s in mod.sections:
            sect_type = s.GetSectionType()
            if sect_type == lldb.eSectionTypeData:
                data_section = s
                break
            elif sect_type == lldb.eSectionTypeContainer:
                for i in range(s.GetNumSubSections()):
                    ss = s.GetSubSectionAtIndex(i)
                    sect_type = ss.GetSectionType()
                    if sect_type == lldb.eSectionTypeData:
                        data_section = ss
                        break                    

        self.assertIsNotNone(data_section)
        return data_section

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
