;; Date:     May 28, 2003.
;; From:     test/Programs/External/SPEC/CINT2000/175.vpr.llvm.bc
;; Function: int %main(int %argc.1, sbyte** %argv.1)
;;
;; Error:    A function call with about 56 arguments causes an assertion failure
;;	     in llc because the register allocator cannot find a register
;;	     not used explicitly by the call instruction.
;; 
;; Cause:    Regalloc was not keeping track of free registers correctly.
;;	     It was counting the registers allocated to all outgoing arguments,
;;	     even though most of those are copied to the stack (so those
;;	     registers are not actually used by the call instruction).
;;
;; Fixed:    By rewriting selection and allocation so that selection explicitly
;;	     inserts all copy operations required for passing arguments and
;;           for the return value of a call, copying to/from registers
;;           and/or to stack locations as needed.
;;

target endian = little
target pointersize = 32
	%struct..s_annealing_sched = type { uint, float, float, float, float }
	%struct..s_chan = type { uint, float, float, float, float }
	%struct..s_det_routing_arch = type { uint, float, float, float, uint, int, short, short, short, float, float }
	%struct..s_placer_opts = type { int, float, int, uint, sbyte*, uint, int }
	%struct..s_router_opts = type { float, float, float, float, float, int, int, uint, int }
	%struct..s_segment_inf = type { float, int, short, short, float, float, uint, float, float }
	%struct..s_switch_inf = type { uint, float, float, float, float }

implementation

int %main(int %argc.1, sbyte** %argv.1) {
entry:		
	%net_file = alloca [300 x sbyte]		
	%place_file = alloca [300 x sbyte]		
	%arch_file = alloca [300 x sbyte]		
	%route_file = alloca [300 x sbyte]		
	%full_stats = alloca uint		
	%operation = alloca int		
	%verify_binary_search = alloca uint		
	%show_graphics = alloca uint		
	%annealing_sched = alloca %struct..s_annealing_sched		
	%placer_opts = alloca %struct..s_placer_opts		
	%router_opts = alloca %struct..s_router_opts		
	%det_routing_arch = alloca %struct..s_det_routing_arch		
	%segment_inf = alloca %struct..s_segment_inf*		
	%timing_inf = alloca { uint, float, float, float, float, float, float, float, float, float, float }		
	%tmp.101 = getelementptr %struct..s_placer_opts* %placer_opts, long 0, ubyte 4		
	%tmp.105 = getelementptr [300 x sbyte]* %net_file, long 0, long 0		
	%tmp.106 = getelementptr [300 x sbyte]* %arch_file, long 0, long 0		
	%tmp.107 = getelementptr [300 x sbyte]* %place_file, long 0, long 0		
	%tmp.108 = getelementptr [300 x sbyte]* %route_file, long 0, long 0		
	%tmp.109 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 0		
	%tmp.112 = getelementptr %struct..s_placer_opts* %placer_opts, long 0, ubyte 0		
	%tmp.114 = getelementptr %struct..s_placer_opts* %placer_opts, long 0, ubyte 6		
	%tmp.118 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 7		
	%tmp.135 = load int* %operation		
	%tmp.137 = load int* %tmp.112		
	%tmp.138 = getelementptr %struct..s_placer_opts* %placer_opts, long 0, ubyte 1		
	%tmp.139 = load float* %tmp.138		
	%tmp.140 = getelementptr %struct..s_placer_opts* %placer_opts, long 0, ubyte 2		
	%tmp.141 = load int* %tmp.140		
	%tmp.142 = getelementptr %struct..s_placer_opts* %placer_opts, long 0, ubyte 3		
	%tmp.143 = load uint* %tmp.142		
	%tmp.145 = load sbyte** %tmp.101		
	%tmp.146 = getelementptr %struct..s_placer_opts* %placer_opts, long 0, ubyte 5		
	%tmp.147 = load uint* %tmp.146		
	%tmp.149 = load int* %tmp.114		
	%tmp.154 = load uint* %full_stats		
	%tmp.155 = load uint* %verify_binary_search		
	%tmp.156 = getelementptr %struct..s_annealing_sched* %annealing_sched, long 0, ubyte 0		
	%tmp.157 = load uint* %tmp.156		
	%tmp.158 = getelementptr %struct..s_annealing_sched* %annealing_sched, long 0, ubyte 1		
	%tmp.159 = load float* %tmp.158		
	%tmp.160 = getelementptr %struct..s_annealing_sched* %annealing_sched, long 0, ubyte 2		
	%tmp.161 = load float* %tmp.160		
	%tmp.162 = getelementptr %struct..s_annealing_sched* %annealing_sched, long 0, ubyte 3		
	%tmp.163 = load float* %tmp.162		
	%tmp.164 = getelementptr %struct..s_annealing_sched* %annealing_sched, long 0, ubyte 4		
	%tmp.165 = load float* %tmp.164		
	%tmp.166 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 0		
	%tmp.167 = load float* %tmp.166		
	%tmp.168 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 1		
	%tmp.169 = load float* %tmp.168		
	%tmp.170 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 2		
	%tmp.171 = load float* %tmp.170		
	%tmp.172 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 3		
	%tmp.173 = load float* %tmp.172		
	%tmp.174 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 4		
	%tmp.175 = load float* %tmp.174		
	%tmp.176 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 5		
	%tmp.177 = load int* %tmp.176		
	%tmp.178 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 6		
	%tmp.179 = load int* %tmp.178		
	%tmp.181 = load uint* %tmp.118		
	%tmp.182 = getelementptr %struct..s_router_opts* %router_opts, long 0, ubyte 8		
	%tmp.183 = load int* %tmp.182		
	%tmp.184 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 0		
	%tmp.185 = load uint* %tmp.184		
	%tmp.186 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 1		
	%tmp.187 = load float* %tmp.186		
	%tmp.188 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 2		
	%tmp.189 = load float* %tmp.188		
	%tmp.190 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 3		
	%tmp.191 = load float* %tmp.190		
	%tmp.192 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 4		
	%tmp.193 = load uint* %tmp.192		
	%tmp.194 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 5		
	%tmp.195 = load int* %tmp.194		
	%tmp.196 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 6		
	%tmp.197 = load short* %tmp.196		
	%tmp.198 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 7		
	%tmp.199 = load short* %tmp.198		
	%tmp.200 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 8		
	%tmp.201 = load short* %tmp.200		
	%tmp.202 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 9		
	%tmp.203 = load float* %tmp.202		
	%tmp.204 = getelementptr %struct..s_det_routing_arch* %det_routing_arch, long 0, ubyte 10		
	%tmp.205 = load float* %tmp.204		
	%tmp.206 = load %struct..s_segment_inf** %segment_inf		
	%tmp.208 = load uint* %tmp.109		
	%tmp.209 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 1		
	%tmp.210 = load float* %tmp.209		
	%tmp.211 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 2		
	%tmp.212 = load float* %tmp.211		
	%tmp.213 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 3		
	%tmp.214 = load float* %tmp.213		
	%tmp.215 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 4		
	%tmp.216 = load float* %tmp.215		
	%tmp.217 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 5		
	%tmp.218 = load float* %tmp.217		
	%tmp.219 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 6		
	%tmp.220 = load float* %tmp.219		
	%tmp.221 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 7		
	%tmp.222 = load float* %tmp.221		
	%tmp.223 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 8		
	%tmp.224 = load float* %tmp.223		
	%tmp.225 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 9		
	%tmp.226 = load float* %tmp.225		
	%tmp.227 = getelementptr { uint, float, float, float, float, float, float, float, float, float, float }* %timing_inf, long 0, ubyte 10		
	%tmp.228 = load float* %tmp.227		
	call void %place_and_route( int %tmp.135, int %tmp.137, float %tmp.139, int %tmp.141, uint %tmp.143, sbyte* %tmp.145, uint %tmp.147, int %tmp.149, sbyte* %tmp.107, sbyte* %tmp.105, sbyte* %tmp.106, sbyte* %tmp.108, uint %tmp.154, uint %tmp.155, uint %tmp.157, float %tmp.159, float %tmp.161, float %tmp.163, float %tmp.165, float %tmp.167, float %tmp.169, float %tmp.171, float %tmp.173, float %tmp.175, int %tmp.177, int %tmp.179, uint %tmp.181, int %tmp.183, uint %tmp.185, float %tmp.187, float %tmp.189, float %tmp.191, uint %tmp.193, int %tmp.195, short %tmp.197, short %tmp.199, short %tmp.201, float %tmp.203, float %tmp.205, %struct..s_segment_inf* %tmp.206, uint %tmp.208, float %tmp.210, float %tmp.212, float %tmp.214, float %tmp.216, float %tmp.218, float %tmp.220, float %tmp.222, float %tmp.224, float %tmp.226, float %tmp.228 )
	%tmp.231 = load uint* %show_graphics		
	%tmp.232 = setne uint %tmp.231, 0		
	br bool %tmp.232, label %then.2, label %endif.2

then.2:		
	br label %endif.2

endif.2:
	ret int 0
}

declare int %printf(sbyte*, ...)

declare void %place_and_route(int, int, float, int, uint, sbyte*, uint, int, sbyte*, sbyte*, sbyte*, sbyte*, uint, uint, uint, float, float, float, float, float, float, float, float, float, int, int, uint, int, uint, float, float, float, uint, int, short, short, short, float, float, %struct..s_segment_inf*, uint, float, float, float, float, float, float, float, float, float, float)
