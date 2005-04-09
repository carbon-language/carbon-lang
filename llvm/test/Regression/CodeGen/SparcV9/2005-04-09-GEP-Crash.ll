; RUN: llvm-as < %s | llc -march=sparcv9


%x_ = external global { [2530944 x double], [2197000 x double], [2530944 x double], [4 x double], [4 x double], [7 x long], [7 x long] }

implementation

void %norm2u3_() {
entry:
	%tmp.47 = getelementptr [2530944 x double]* cast (double* getelementptr ([2530944 x double]* getelementptr ({ [2530944 x double], [2197000 x double], [2530944 x double], [4 x double], [4 x double], [7 x long], [7 x long] }* %x_, long 0, uint 0), long 1, long 2197000) to [2530944 x double]*), long 0, long 0
	ret void
}
