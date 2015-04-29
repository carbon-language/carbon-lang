
import os
import commands

perl = "/usr/bin/perl"
LLVM = "./LLVM-IR/"
temp_filename = "temp"
XFAIL_text = "; XFAIL: *\n"


arch_file_list = dict()
arch_file_list['lin_32e'] = ['test_omp_task_final.ll', 'test_omp_task_untied.ll']


arch_script = "../runtime/tools/check-openmp-test.pl"
arch_cmd = perl + " " + arch_script
arch = commands.getoutput(arch_cmd)
arch = arch[:len(arch)-1]

print "Adding XFAILS ..."

for f in arch_file_list[arch]:
	filename = LLVM + arch + "/" + f
	lines = open(filename).readlines()
	lines.insert(1, XFAIL_text)
	f2 = open(temp_filename, "w")
	for l in lines:
		f2.write(l)
	f2.close()

	os.system("mv " + temp_filename + " " + filename)

