"""
Test SBSection APIs.
"""



from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SectionAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_get_target_byte_size(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = self.getBuildArtifact('b.out')
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
        self.assertEqual(data_section.target_byte_size, 1)
