"""
Test SBSection APIs.
"""

import unittest2
from lldbtest import *

class SectionAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @python_api_test
    def test_get_target_byte_size(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = os.path.join(os.getcwd(), 'b.out')
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # find the .data section of the main module            
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
        self.assertEquals(data_section.target_byte_size, 1)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
