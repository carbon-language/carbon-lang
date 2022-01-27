import bz2
import shutil
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfFBSDVMCoreSupportMissing
class FreeBSDKernelVMCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    maxDiff = None

    mydir = TestBase.compute_mydir(__file__)

    def make_target(self, src_filename):
        src = self.getSourcePath(src_filename)
        dest = self.getBuildArtifact("kernel")
        self.yaml2obj(src, dest, max_size=30*1024*1024)
        return self.dbg.CreateTarget(dest)

    def make_vmcore(self, src_filename):
        src = self.getSourcePath(src_filename)
        dest = self.getBuildArtifact("vmcore")
        with bz2.open(src, "rb") as inf:
            with open(dest, "wb") as outf:
                shutil.copyfileobj(inf, outf)
        return dest

    def do_test(self, kernel_yaml, vmcore_bz2, numthread, threads={}, hz=100):
        target = self.make_target(kernel_yaml)
        vmcore_file = self.make_vmcore(vmcore_bz2)
        process = target.LoadCore(vmcore_file)

        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), numthread)
        self.assertEqual(process.GetProcessID(), 0)

        # test memory reading
        self.expect("expr -- *(int *) &hz",
                    substrs=["(int) $0 = %d" % hz])

        main_mod = target.GetModuleAtIndex(0)
        hz_addr = (main_mod.FindSymbols("hz")[0].symbol.addr
                   .GetLoadAddress(target))
        error = lldb.SBError()
        self.assertEqual(process.ReadMemory(hz_addr, 4, error),
                         struct.pack("<I", hz))

        for thread_index, thread_data in threads.items():
            bt_expected = thread_data["bt"]
            regs_expected = thread_data["regs"]
            thread = process.GetThreadAtIndex(thread_index)
            self.assertEqual(thread.GetName(), thread_data["name"])

            # test backtrace
            self.assertEqual(
                [frame.addr.GetLoadAddress(target) for frame in thread],
                bt_expected)

            # test registers
            regs = thread.GetFrameAtIndex(0).GetRegisters()
            reg_values = {}
            for regset in regs:
                for reg in regset:
                    if reg.value is None:
                        continue
                    reg_values[reg.name] = reg.value
            self.assertEqual(reg_values, regs_expected)

        self.dbg.DeleteTarget(target)

    def test_amd64_full_vmcore(self):
        self.do_test("kernel-amd64.yaml", "vmcore-amd64-full.bz2",
                numthread=13,
                threads={
                    0: {"name": "(pid 806) sysctl (crashed)",
                        "bt": [0xffffffff80c09ade, 0xffffffff80c09916,
                               0xffffffff80c09d90, 0xffffffff80c09b93,
                               0xffffffff80c57d91, 0xffffffff80c19e71,
                               0xffffffff80c192bc, 0xffffffff80c19933,
                               0xffffffff80c1977f, 0xffffffff8108ba8c,
                               0xffffffff810620ce,
                               ],
                        "regs": {"rbx": "0x0000000000000000",
                                 "rbp": "0xfffffe0085cb2760",
                                 "rsp": "0xfffffe0085cb2748",
                                 "r12": "0xfffffe0045a6c300",
                                 "r13": "0xfffff800033693a8",
                                 "r14": "0x0000000000000000",
                                 "r15": "0xfffff80003369380",
                                 "rip": "0xffffffff80c09ade",
                                 },
                        },
                    1: {"name": "(pid 11) idle/idle: cpu0 (on CPU 0)",
                        "bt": [0xffffffff81057988, 0xffffffff81057949,
                               0xffffffff8108a5ff, 0xffffffff81062537,
                               0xffffffff80c1aa38, 0xffffffff80b9d587,
                               0xffffffff810575b1, 0xffffffff81063b33,
                               0xffffffff804e3edb, 0xffffffff8104dc6e,
                               0xffffffff8104dd1f, 0xffffffff80c3f0b4,
                               0xffffffff80bc7c5e,
                               ],
                        "regs": {"rbx": "0xffffffff81d43950",
                                 "rbp": "0xffffffff81d43820",
                                 "rsp": "0xffffffff81d43808",
                                 "r12": "0xfffff80003374000",
                                 "r13": "0x00000000027be000",
                                 "r14": "0x0000000000000000",
                                 "r15": "0xfffffe00009f7300",
                                 "rip": "0xffffffff81057988",
                                 },
                        },
                    10: {"name": "(pid 11) idle/idle: cpu9",
                         "bt": [0xffffffff80c3c8c8, 0xffffffff80c16521,
                                0xffffffff80c3f0b4, 0xffffffff80bc7c5e,
                                ],
                         "regs": {"rbx": "0x000000007fff29f4",
                                  "rbp": "0xfffffe00007a4ad0",
                                  "rsp": "0xfffffe00007a4a08",
                                  "r12": "0xfffffe00009fd300",
                                  "r13": "0x0000000000000608",
                                  "r14": "0xfffffe00009250c0",
                                  "r15": "0xfffffe0045a6c300",
                                  "rip": "0xffffffff80c3c8c8",
                                  },
                         },
                    })

    def test_amd64_minidump(self):
        self.do_test("kernel-amd64.yaml", "vmcore-amd64-minidump.bz2",
                     numthread=16,
                     threads={
                         0: {"name": "(pid 800) sysctl (crashed)",
                             "bt": [0xffffffff80c09ade, 0xffffffff80c09916,
                                    0xffffffff80c09d90, 0xffffffff80c09b93,
                                    0xffffffff80c57d91, 0xffffffff80c19e71,
                                    0xffffffff80c192bc, 0xffffffff80c19933,
                                    0xffffffff80c1977f, 0xffffffff8108ba8c,
                                    0xffffffff810620ce,
                                    ],
                             "regs": {"rbx": "0x0000000000000000",
                                      "rbp": "0xfffffe00798c4760",
                                      "rsp": "0xfffffe00798c4748",
                                      "r12": "0xfffffe0045b11c00",
                                      "r13": "0xfffff800033693a8",
                                      "r14": "0x0000000000000000",
                                      "r15": "0xfffff80003369380",
                                      "rip": "0xffffffff80c09ade",
                                      },
                             },
                         1: {"name": "(pid 28) pagedaemon/dom0 (on CPU 4)",
                             "bt": [0xffffffff81057988, 0xffffffff81057949,
                                    0xffffffff8108a5ff, 0xffffffff81062537,
                                    0xffffffff8107171e, 0xffffffff81075f9c,
                                    0xffffffff80f4359e, 0xffffffff80f494b4,
                                    0xffffffff80f47430, 0xffffffff80f46eee,
                                    0xffffffff80bc7c5e,
                                    ],
                             "regs": {"rbx": "0xfffffe00008e2f30",
                                      "rbp": "0xfffffe00008e2e00",
                                      "rsp": "0xfffffe00008e2de8",
                                      "r12": "0xfffff80003845000",
                                      "r13": "0x00000000027be000",
                                      "r14": "0x0000000000000004",
                                      "r15": "0xfffffe00458c2700",
                                      "rip": "0xffffffff81057988",
                                      },
                             },
                         2: {"name": "(pid 28) pagedaemon/laundry: dom0",
                             "bt": [0xffffffff80c3c8c8, 0xffffffff80c16521,
                                    0xffffffff80c15c3b, 0xffffffff80f48dfc,
                                    0xffffffff80bc7c5e,
                                    ],
                             "regs": {"rbx": "0x000000007fff25f1",
                                      "rbp": "0xfffffe00527dd890",
                                      "rsp": "0xfffffe00527dd7c8",
                                      "r12": "0xfffffe0045b13100",
                                      "r13": "0x0000000000000104",
                                      "r14": "0xfffffe00008d70c0",
                                      "r15": "0xfffffe00009f5e00",
                                      "rip": "0xffffffff80c3c8c8",
                                      },
                             },
                         })

    def test_arm64_minidump(self):
        self.do_test("kernel-arm64.yaml", "vmcore-arm64-minidump.bz2",
                     hz=1000,
                     numthread=10,
                     threads={
                         0: {"name": "(pid 939) sysctl (crashed)",
                             # TODO: fix unwinding
                             "bt": [0xffff0000004b6e78,
                                    ],
                             "regs": {"x0": "0x0000000000000000",
                                      "x1": "0x0000000000000000",
                                      "x2": "0x0000000000000000",
                                      "x3": "0x0000000000000000",
                                      "x4": "0x0000000000000000",
                                      "x5": "0x0000000000000000",
                                      "x6": "0x0000000000000000",
                                      "x7": "0x0000000000000000",
                                      "x8": "0xffffa00001548700",
                                      "x9": "0x0000000000000000",
                                      "x10": "0xffffa00000e04580",
                                      "x11": "0x0000000000000000",
                                      "x12": "0x000000000008950a",
                                      "x13": "0x0000000000089500",
                                      "x14": "0x0000000000000039",
                                      "x15": "0x0000000000000000",
                                      "x16": "0x00000000ffffffd8",
                                      "x17": "0x0000000000000000",
                                      "x18": "0xffff000000e6d380",
                                      "x19": "0xffff000000af9000",
                                      "x20": "0xffff000000b82000",
                                      "x21": "0xffffa00000319da8",
                                      "x22": "0xffff000000b84000",
                                      "x23": "0xffff000000b84000",
                                      "x24": "0xffff000000b55000",
                                      "x25": "0x0000000000000000",
                                      "x26": "0x0000000000040800",
                                      "x27": "0x0000000000000000",
                                      "x28": "0x00000000002019ca",
                                      "fp": "0xffff0000d58f23b0",
                                      "sp": "0xffff0000d58f23b0",
                                      "pc": "0xffff0000004b6e78",
                                      },
                             },
                         1: {"name": "(pid 21) syncer (on CPU 6)",
                             # TODO: fix unwinding
                             "bt": [0xffff000000811370,
                                    ],
                             "regs": {"x0": "0x0000000000000000",
                                      "x1": "0x0000000000000000",
                                      "x2": "0x0000000000000000",
                                      "x3": "0x0000000000000000",
                                      "x4": "0x0000000000000000",
                                      "x5": "0x0000000000000000",
                                      "x6": "0x0000000000000000",
                                      "x7": "0x0000000000000000",
                                      "x8": "0x0000000000000006",
                                      "x9": "0x0000000000000560",
                                      "x10": "0xffff000000e8f640",
                                      "x11": "0x0000000000000001",
                                      "x12": "0x0000000056000000",
                                      "x13": "0x0000000000002af8",
                                      "x14": "0x0000000000002710",
                                      "x15": "0x0000000000000002",
                                      "x16": "0x00000000ffffffff",
                                      "x17": "0x0000000000000002",
                                      "x18": "0xffff000000e6db80",
                                      "x19": "0x0000000000000006",
                                      "x20": "0xffff0000853a3670",
                                      "x21": "0xffff0000009279c1",
                                      "x22": "0x0000000000000804",
                                      "x23": "0x0000000000000004",
                                      "x24": "0xffff000082a93000",
                                      "x25": "0xffffa0000053f080",
                                      "x26": "0xffff000000e6391c",
                                      "x27": "0xffff000000e63000",
                                      "x28": "0x0000000000000004",
                                      "fp": "0xffff0000853a35c0",
                                      "sp": "0xffff0000853a35c0",
                                      "pc": "0xffff000000811370",
                                      },
                             },
                         4: {"name": "(pid 11) idle/idle: cpu2",
                             # TODO: fix unwinding
                             "bt": [0xffff0000004ee99c,
                                    ],
                             "regs": {"x0": "0x0000000000000000",
                                      "x1": "0x0000000000000000",
                                      "x2": "0x0000000000000000",
                                      "x3": "0x0000000000000000",
                                      "x4": "0x0000000000000000",
                                      "x5": "0x0000000000000000",
                                      "x6": "0x0000000000000000",
                                      "x7": "0x0000000000000000",
                                      "x8": "0x00000000ffffffff",
                                      "x9": "0x0000000000000001",
                                      "x10": "0x0000000000002710",
                                      "x11": "0x000000007ff7e333",
                                      "x12": "0x000000007ff7ba9c",
                                      "x13": "0x0000000000002af8",
                                      "x14": "0x0000000000002897",
                                      "x15": "0x0000000000002af8",
                                      "x16": "0x00000000000028e1",
                                      "x17": "0x00000000ffffffff",
                                      "x18": "0xffff000000e6d380",
                                      "x19": "0xffffa0000032e580",
                                      "x20": "0xffff000000b82000",
                                      "x21": "0xffff000040517100",
                                      "x22": "0xffffa00000e04580",
                                      "x23": "0xffff000000b84000",
                                      "x24": "0x0000000000000001",
                                      "x25": "0xffff000000dd1000",
                                      "x26": "0xffff000082783898",
                                      "x27": "0xffff000000e26000",
                                      "x28": "0xffff000000b82000",
                                      "fp": "0xffff0000827835f0",
                                      "sp": "0xffff000082783570",
                                      "pc": "0xffff0000004ee99c",
                                      },
                             },
                         })

    def test_i386_minidump(self):
        self.do_test("kernel-i386.yaml", "vmcore-i386-minidump.bz2",
                     numthread=13,
                     threads={
                         0: {"name": "(pid 806) sysctl (crashed)",
                             "bt": [0x010025c5, 0x01002410,
                                    0x010027d5, 0x01002644,
                                    0x01049a2f, 0x01011077,
                                    0x01010780, 0x01010c7a,
                                    0x01010ab2, 0x013e9e2d,
                                    0xffc033f9,
                                    ],
                             "regs": {"ebp": "0x151968e4",
                                      "esp": "0x151968d8",
                                      "esi": "0x0c77aa80",
                                      "edi": "0x03f0dc80",
                                      "eip": "0x010025c5",
                                      },
                             },
                         1: {"name": "(pid 11) idle/idle: cpu0 (on CPU 0)",
                             "bt": [0x013a91f6, 0x013a91c0,
                                    0x013e8ce4, 0xffc0319f,
                                    0x00000028,
                                    ],
                             "regs": {"ebp": "0x03d979bc",
                                      "esp": "0x03d979a0",
                                      "esi": "0x000007f7",
                                      "edi": "0x00000000",
                                      "eip": "0x013a91f6",
                                      },
                             },
                         12: {"name": "(pid 11) idle/idle: cpu11",
                             "bt": [0x0103012c, 0x0100de0e,
                                    0x0100b770, 0x010323be,
                                    0x00fc50b6,
                                    ],
                             "regs": {"ebp": "0x03dc4af0",
                                      "esp": "0x03dc4aa4",
                                      "esi": "0x03f97e00",
                                      "edi": "0x000003e8",
                                      "eip": "0x0103012c",
                                      },
                             },
                         })
