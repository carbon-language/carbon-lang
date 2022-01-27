; -enable-misched=false makes the register usage more predictable
; -regalloc=fast just makes the test run faster
; RUN: llc -march=amdgcn -mcpu=gfx90a -amdgpu-function-calls=false -enable-misched=false -sgpr-regalloc=fast -vgpr-regalloc=fast < %s | FileCheck %s --check-prefixes=GCN,GFX90A

define internal void @use256vgprs() {
  %v0 = call i32 asm sideeffect "; def $0", "=v"()
  %v1 = call i32 asm sideeffect "; def $0", "=v"()
  %v2 = call i32 asm sideeffect "; def $0", "=v"()
  %v3 = call i32 asm sideeffect "; def $0", "=v"()
  %v4 = call i32 asm sideeffect "; def $0", "=v"()
  %v5 = call i32 asm sideeffect "; def $0", "=v"()
  %v6 = call i32 asm sideeffect "; def $0", "=v"()
  %v7 = call i32 asm sideeffect "; def $0", "=v"()
  %v8 = call i32 asm sideeffect "; def $0", "=v"()
  %v9 = call i32 asm sideeffect "; def $0", "=v"()
  %v10 = call i32 asm sideeffect "; def $0", "=v"()
  %v11 = call i32 asm sideeffect "; def $0", "=v"()
  %v12 = call i32 asm sideeffect "; def $0", "=v"()
  %v13 = call i32 asm sideeffect "; def $0", "=v"()
  %v14 = call i32 asm sideeffect "; def $0", "=v"()
  %v15 = call i32 asm sideeffect "; def $0", "=v"()
  %v16 = call i32 asm sideeffect "; def $0", "=v"()
  %v17 = call i32 asm sideeffect "; def $0", "=v"()
  %v18 = call i32 asm sideeffect "; def $0", "=v"()
  %v19 = call i32 asm sideeffect "; def $0", "=v"()
  %v20 = call i32 asm sideeffect "; def $0", "=v"()
  %v21 = call i32 asm sideeffect "; def $0", "=v"()
  %v22 = call i32 asm sideeffect "; def $0", "=v"()
  %v23 = call i32 asm sideeffect "; def $0", "=v"()
  %v24 = call i32 asm sideeffect "; def $0", "=v"()
  %v25 = call i32 asm sideeffect "; def $0", "=v"()
  %v26 = call i32 asm sideeffect "; def $0", "=v"()
  %v27 = call i32 asm sideeffect "; def $0", "=v"()
  %v28 = call i32 asm sideeffect "; def $0", "=v"()
  %v29 = call i32 asm sideeffect "; def $0", "=v"()
  %v30 = call i32 asm sideeffect "; def $0", "=v"()
  %v31 = call i32 asm sideeffect "; def $0", "=v"()
  %v32 = call i32 asm sideeffect "; def $0", "=v"()
  %v33 = call i32 asm sideeffect "; def $0", "=v"()
  %v34 = call i32 asm sideeffect "; def $0", "=v"()
  %v35 = call i32 asm sideeffect "; def $0", "=v"()
  %v36 = call i32 asm sideeffect "; def $0", "=v"()
  %v37 = call i32 asm sideeffect "; def $0", "=v"()
  %v38 = call i32 asm sideeffect "; def $0", "=v"()
  %v39 = call i32 asm sideeffect "; def $0", "=v"()
  %v40 = call i32 asm sideeffect "; def $0", "=v"()
  %v41 = call i32 asm sideeffect "; def $0", "=v"()
  %v42 = call i32 asm sideeffect "; def $0", "=v"()
  %v43 = call i32 asm sideeffect "; def $0", "=v"()
  %v44 = call i32 asm sideeffect "; def $0", "=v"()
  %v45 = call i32 asm sideeffect "; def $0", "=v"()
  %v46 = call i32 asm sideeffect "; def $0", "=v"()
  %v47 = call i32 asm sideeffect "; def $0", "=v"()
  %v48 = call i32 asm sideeffect "; def $0", "=v"()
  %v49 = call i32 asm sideeffect "; def $0", "=v"()
  %v50 = call i32 asm sideeffect "; def $0", "=v"()
  %v51 = call i32 asm sideeffect "; def $0", "=v"()
  %v52 = call i32 asm sideeffect "; def $0", "=v"()
  %v53 = call i32 asm sideeffect "; def $0", "=v"()
  %v54 = call i32 asm sideeffect "; def $0", "=v"()
  %v55 = call i32 asm sideeffect "; def $0", "=v"()
  %v56 = call i32 asm sideeffect "; def $0", "=v"()
  %v57 = call i32 asm sideeffect "; def $0", "=v"()
  %v58 = call i32 asm sideeffect "; def $0", "=v"()
  %v59 = call i32 asm sideeffect "; def $0", "=v"()
  %v60 = call i32 asm sideeffect "; def $0", "=v"()
  %v61 = call i32 asm sideeffect "; def $0", "=v"()
  %v62 = call i32 asm sideeffect "; def $0", "=v"()
  %v63 = call i32 asm sideeffect "; def $0", "=v"()
  %v64 = call i32 asm sideeffect "; def $0", "=v"()
  %v65 = call i32 asm sideeffect "; def $0", "=v"()
  %v66 = call i32 asm sideeffect "; def $0", "=v"()
  %v67 = call i32 asm sideeffect "; def $0", "=v"()
  %v68 = call i32 asm sideeffect "; def $0", "=v"()
  %v69 = call i32 asm sideeffect "; def $0", "=v"()
  %v70 = call i32 asm sideeffect "; def $0", "=v"()
  %v71 = call i32 asm sideeffect "; def $0", "=v"()
  %v72 = call i32 asm sideeffect "; def $0", "=v"()
  %v73 = call i32 asm sideeffect "; def $0", "=v"()
  %v74 = call i32 asm sideeffect "; def $0", "=v"()
  %v75 = call i32 asm sideeffect "; def $0", "=v"()
  %v76 = call i32 asm sideeffect "; def $0", "=v"()
  %v77 = call i32 asm sideeffect "; def $0", "=v"()
  %v78 = call i32 asm sideeffect "; def $0", "=v"()
  %v79 = call i32 asm sideeffect "; def $0", "=v"()
  %v80 = call i32 asm sideeffect "; def $0", "=v"()
  %v81 = call i32 asm sideeffect "; def $0", "=v"()
  %v82 = call i32 asm sideeffect "; def $0", "=v"()
  %v83 = call i32 asm sideeffect "; def $0", "=v"()
  %v84 = call i32 asm sideeffect "; def $0", "=v"()
  %v85 = call i32 asm sideeffect "; def $0", "=v"()
  %v86 = call i32 asm sideeffect "; def $0", "=v"()
  %v87 = call i32 asm sideeffect "; def $0", "=v"()
  %v88 = call i32 asm sideeffect "; def $0", "=v"()
  %v89 = call i32 asm sideeffect "; def $0", "=v"()
  %v90 = call i32 asm sideeffect "; def $0", "=v"()
  %v91 = call i32 asm sideeffect "; def $0", "=v"()
  %v92 = call i32 asm sideeffect "; def $0", "=v"()
  %v93 = call i32 asm sideeffect "; def $0", "=v"()
  %v94 = call i32 asm sideeffect "; def $0", "=v"()
  %v95 = call i32 asm sideeffect "; def $0", "=v"()
  %v96 = call i32 asm sideeffect "; def $0", "=v"()
  %v97 = call i32 asm sideeffect "; def $0", "=v"()
  %v98 = call i32 asm sideeffect "; def $0", "=v"()
  %v99 = call i32 asm sideeffect "; def $0", "=v"()
  %v100 = call i32 asm sideeffect "; def $0", "=v"()
  %v101 = call i32 asm sideeffect "; def $0", "=v"()
  %v102 = call i32 asm sideeffect "; def $0", "=v"()
  %v103 = call i32 asm sideeffect "; def $0", "=v"()
  %v104 = call i32 asm sideeffect "; def $0", "=v"()
  %v105 = call i32 asm sideeffect "; def $0", "=v"()
  %v106 = call i32 asm sideeffect "; def $0", "=v"()
  %v107 = call i32 asm sideeffect "; def $0", "=v"()
  %v108 = call i32 asm sideeffect "; def $0", "=v"()
  %v109 = call i32 asm sideeffect "; def $0", "=v"()
  %v110 = call i32 asm sideeffect "; def $0", "=v"()
  %v111 = call i32 asm sideeffect "; def $0", "=v"()
  %v112 = call i32 asm sideeffect "; def $0", "=v"()
  %v113 = call i32 asm sideeffect "; def $0", "=v"()
  %v114 = call i32 asm sideeffect "; def $0", "=v"()
  %v115 = call i32 asm sideeffect "; def $0", "=v"()
  %v116 = call i32 asm sideeffect "; def $0", "=v"()
  %v117 = call i32 asm sideeffect "; def $0", "=v"()
  %v118 = call i32 asm sideeffect "; def $0", "=v"()
  %v119 = call i32 asm sideeffect "; def $0", "=v"()
  %v120 = call i32 asm sideeffect "; def $0", "=v"()
  %v121 = call i32 asm sideeffect "; def $0", "=v"()
  %v122 = call i32 asm sideeffect "; def $0", "=v"()
  %v123 = call i32 asm sideeffect "; def $0", "=v"()
  %v124 = call i32 asm sideeffect "; def $0", "=v"()
  %v125 = call i32 asm sideeffect "; def $0", "=v"()
  %v126 = call i32 asm sideeffect "; def $0", "=v"()
  %v127 = call i32 asm sideeffect "; def $0", "=v"()
  %v128 = call i32 asm sideeffect "; def $0", "=v"()
  %v129 = call i32 asm sideeffect "; def $0", "=v"()
  %v130 = call i32 asm sideeffect "; def $0", "=v"()
  %v131 = call i32 asm sideeffect "; def $0", "=v"()
  %v132 = call i32 asm sideeffect "; def $0", "=v"()
  %v133 = call i32 asm sideeffect "; def $0", "=v"()
  %v134 = call i32 asm sideeffect "; def $0", "=v"()
  %v135 = call i32 asm sideeffect "; def $0", "=v"()
  %v136 = call i32 asm sideeffect "; def $0", "=v"()
  %v137 = call i32 asm sideeffect "; def $0", "=v"()
  %v138 = call i32 asm sideeffect "; def $0", "=v"()
  %v139 = call i32 asm sideeffect "; def $0", "=v"()
  %v140 = call i32 asm sideeffect "; def $0", "=v"()
  %v141 = call i32 asm sideeffect "; def $0", "=v"()
  %v142 = call i32 asm sideeffect "; def $0", "=v"()
  %v143 = call i32 asm sideeffect "; def $0", "=v"()
  %v144 = call i32 asm sideeffect "; def $0", "=v"()
  %v145 = call i32 asm sideeffect "; def $0", "=v"()
  %v146 = call i32 asm sideeffect "; def $0", "=v"()
  %v147 = call i32 asm sideeffect "; def $0", "=v"()
  %v148 = call i32 asm sideeffect "; def $0", "=v"()
  %v149 = call i32 asm sideeffect "; def $0", "=v"()
  %v150 = call i32 asm sideeffect "; def $0", "=v"()
  %v151 = call i32 asm sideeffect "; def $0", "=v"()
  %v152 = call i32 asm sideeffect "; def $0", "=v"()
  %v153 = call i32 asm sideeffect "; def $0", "=v"()
  %v154 = call i32 asm sideeffect "; def $0", "=v"()
  %v155 = call i32 asm sideeffect "; def $0", "=v"()
  %v156 = call i32 asm sideeffect "; def $0", "=v"()
  %v157 = call i32 asm sideeffect "; def $0", "=v"()
  %v158 = call i32 asm sideeffect "; def $0", "=v"()
  %v159 = call i32 asm sideeffect "; def $0", "=v"()
  %v160 = call i32 asm sideeffect "; def $0", "=v"()
  %v161 = call i32 asm sideeffect "; def $0", "=v"()
  %v162 = call i32 asm sideeffect "; def $0", "=v"()
  %v163 = call i32 asm sideeffect "; def $0", "=v"()
  %v164 = call i32 asm sideeffect "; def $0", "=v"()
  %v165 = call i32 asm sideeffect "; def $0", "=v"()
  %v166 = call i32 asm sideeffect "; def $0", "=v"()
  %v167 = call i32 asm sideeffect "; def $0", "=v"()
  %v168 = call i32 asm sideeffect "; def $0", "=v"()
  %v169 = call i32 asm sideeffect "; def $0", "=v"()
  %v170 = call i32 asm sideeffect "; def $0", "=v"()
  %v171 = call i32 asm sideeffect "; def $0", "=v"()
  %v172 = call i32 asm sideeffect "; def $0", "=v"()
  %v173 = call i32 asm sideeffect "; def $0", "=v"()
  %v174 = call i32 asm sideeffect "; def $0", "=v"()
  %v175 = call i32 asm sideeffect "; def $0", "=v"()
  %v176 = call i32 asm sideeffect "; def $0", "=v"()
  %v177 = call i32 asm sideeffect "; def $0", "=v"()
  %v178 = call i32 asm sideeffect "; def $0", "=v"()
  %v179 = call i32 asm sideeffect "; def $0", "=v"()
  %v180 = call i32 asm sideeffect "; def $0", "=v"()
  %v181 = call i32 asm sideeffect "; def $0", "=v"()
  %v182 = call i32 asm sideeffect "; def $0", "=v"()
  %v183 = call i32 asm sideeffect "; def $0", "=v"()
  %v184 = call i32 asm sideeffect "; def $0", "=v"()
  %v185 = call i32 asm sideeffect "; def $0", "=v"()
  %v186 = call i32 asm sideeffect "; def $0", "=v"()
  %v187 = call i32 asm sideeffect "; def $0", "=v"()
  %v188 = call i32 asm sideeffect "; def $0", "=v"()
  %v189 = call i32 asm sideeffect "; def $0", "=v"()
  %v190 = call i32 asm sideeffect "; def $0", "=v"()
  %v191 = call i32 asm sideeffect "; def $0", "=v"()
  %v192 = call i32 asm sideeffect "; def $0", "=v"()
  %v193 = call i32 asm sideeffect "; def $0", "=v"()
  %v194 = call i32 asm sideeffect "; def $0", "=v"()
  %v195 = call i32 asm sideeffect "; def $0", "=v"()
  %v196 = call i32 asm sideeffect "; def $0", "=v"()
  %v197 = call i32 asm sideeffect "; def $0", "=v"()
  %v198 = call i32 asm sideeffect "; def $0", "=v"()
  %v199 = call i32 asm sideeffect "; def $0", "=v"()
  %v200 = call i32 asm sideeffect "; def $0", "=v"()
  %v201 = call i32 asm sideeffect "; def $0", "=v"()
  %v202 = call i32 asm sideeffect "; def $0", "=v"()
  %v203 = call i32 asm sideeffect "; def $0", "=v"()
  %v204 = call i32 asm sideeffect "; def $0", "=v"()
  %v205 = call i32 asm sideeffect "; def $0", "=v"()
  %v206 = call i32 asm sideeffect "; def $0", "=v"()
  %v207 = call i32 asm sideeffect "; def $0", "=v"()
  %v208 = call i32 asm sideeffect "; def $0", "=v"()
  %v209 = call i32 asm sideeffect "; def $0", "=v"()
  %v210 = call i32 asm sideeffect "; def $0", "=v"()
  %v211 = call i32 asm sideeffect "; def $0", "=v"()
  %v212 = call i32 asm sideeffect "; def $0", "=v"()
  %v213 = call i32 asm sideeffect "; def $0", "=v"()
  %v214 = call i32 asm sideeffect "; def $0", "=v"()
  %v215 = call i32 asm sideeffect "; def $0", "=v"()
  %v216 = call i32 asm sideeffect "; def $0", "=v"()
  %v217 = call i32 asm sideeffect "; def $0", "=v"()
  %v218 = call i32 asm sideeffect "; def $0", "=v"()
  %v219 = call i32 asm sideeffect "; def $0", "=v"()
  %v220 = call i32 asm sideeffect "; def $0", "=v"()
  %v221 = call i32 asm sideeffect "; def $0", "=v"()
  %v222 = call i32 asm sideeffect "; def $0", "=v"()
  %v223 = call i32 asm sideeffect "; def $0", "=v"()
  %v224 = call i32 asm sideeffect "; def $0", "=v"()
  %v225 = call i32 asm sideeffect "; def $0", "=v"()
  %v226 = call i32 asm sideeffect "; def $0", "=v"()
  %v227 = call i32 asm sideeffect "; def $0", "=v"()
  %v228 = call i32 asm sideeffect "; def $0", "=v"()
  %v229 = call i32 asm sideeffect "; def $0", "=v"()
  %v230 = call i32 asm sideeffect "; def $0", "=v"()
  %v231 = call i32 asm sideeffect "; def $0", "=v"()
  %v232 = call i32 asm sideeffect "; def $0", "=v"()
  %v233 = call i32 asm sideeffect "; def $0", "=v"()
  %v234 = call i32 asm sideeffect "; def $0", "=v"()
  %v235 = call i32 asm sideeffect "; def $0", "=v"()
  %v236 = call i32 asm sideeffect "; def $0", "=v"()
  %v237 = call i32 asm sideeffect "; def $0", "=v"()
  %v238 = call i32 asm sideeffect "; def $0", "=v"()
  %v239 = call i32 asm sideeffect "; def $0", "=v"()
  %v240 = call i32 asm sideeffect "; def $0", "=v"()
  %v241 = call i32 asm sideeffect "; def $0", "=v"()
  %v242 = call i32 asm sideeffect "; def $0", "=v"()
  %v243 = call i32 asm sideeffect "; def $0", "=v"()
  %v244 = call i32 asm sideeffect "; def $0", "=v"()
  %v245 = call i32 asm sideeffect "; def $0", "=v"()
  %v246 = call i32 asm sideeffect "; def $0", "=v"()
  %v247 = call i32 asm sideeffect "; def $0", "=v"()
  %v248 = call i32 asm sideeffect "; def $0", "=v"()
  %v249 = call i32 asm sideeffect "; def $0", "=v"()
  %v250 = call i32 asm sideeffect "; def $0", "=v"()
  %v251 = call i32 asm sideeffect "; def $0", "=v"()
  %v252 = call i32 asm sideeffect "; def $0", "=v"()
  %v253 = call i32 asm sideeffect "; def $0", "=v"()
  %v254 = call i32 asm sideeffect "; def $0", "=v"()
  %v255 = call i32 asm sideeffect "; def $0", "=v"()
  call void asm sideeffect "; use $0", "v"(i32 %v0)
  call void asm sideeffect "; use $0", "v"(i32 %v1)
  call void asm sideeffect "; use $0", "v"(i32 %v2)
  call void asm sideeffect "; use $0", "v"(i32 %v3)
  call void asm sideeffect "; use $0", "v"(i32 %v4)
  call void asm sideeffect "; use $0", "v"(i32 %v5)
  call void asm sideeffect "; use $0", "v"(i32 %v6)
  call void asm sideeffect "; use $0", "v"(i32 %v7)
  call void asm sideeffect "; use $0", "v"(i32 %v8)
  call void asm sideeffect "; use $0", "v"(i32 %v9)
  call void asm sideeffect "; use $0", "v"(i32 %v10)
  call void asm sideeffect "; use $0", "v"(i32 %v11)
  call void asm sideeffect "; use $0", "v"(i32 %v12)
  call void asm sideeffect "; use $0", "v"(i32 %v13)
  call void asm sideeffect "; use $0", "v"(i32 %v14)
  call void asm sideeffect "; use $0", "v"(i32 %v15)
  call void asm sideeffect "; use $0", "v"(i32 %v16)
  call void asm sideeffect "; use $0", "v"(i32 %v17)
  call void asm sideeffect "; use $0", "v"(i32 %v18)
  call void asm sideeffect "; use $0", "v"(i32 %v19)
  call void asm sideeffect "; use $0", "v"(i32 %v20)
  call void asm sideeffect "; use $0", "v"(i32 %v21)
  call void asm sideeffect "; use $0", "v"(i32 %v22)
  call void asm sideeffect "; use $0", "v"(i32 %v23)
  call void asm sideeffect "; use $0", "v"(i32 %v24)
  call void asm sideeffect "; use $0", "v"(i32 %v25)
  call void asm sideeffect "; use $0", "v"(i32 %v26)
  call void asm sideeffect "; use $0", "v"(i32 %v27)
  call void asm sideeffect "; use $0", "v"(i32 %v28)
  call void asm sideeffect "; use $0", "v"(i32 %v29)
  call void asm sideeffect "; use $0", "v"(i32 %v30)
  call void asm sideeffect "; use $0", "v"(i32 %v31)
  call void asm sideeffect "; use $0", "v"(i32 %v32)
  call void asm sideeffect "; use $0", "v"(i32 %v33)
  call void asm sideeffect "; use $0", "v"(i32 %v34)
  call void asm sideeffect "; use $0", "v"(i32 %v35)
  call void asm sideeffect "; use $0", "v"(i32 %v36)
  call void asm sideeffect "; use $0", "v"(i32 %v37)
  call void asm sideeffect "; use $0", "v"(i32 %v38)
  call void asm sideeffect "; use $0", "v"(i32 %v39)
  call void asm sideeffect "; use $0", "v"(i32 %v40)
  call void asm sideeffect "; use $0", "v"(i32 %v41)
  call void asm sideeffect "; use $0", "v"(i32 %v42)
  call void asm sideeffect "; use $0", "v"(i32 %v43)
  call void asm sideeffect "; use $0", "v"(i32 %v44)
  call void asm sideeffect "; use $0", "v"(i32 %v45)
  call void asm sideeffect "; use $0", "v"(i32 %v46)
  call void asm sideeffect "; use $0", "v"(i32 %v47)
  call void asm sideeffect "; use $0", "v"(i32 %v48)
  call void asm sideeffect "; use $0", "v"(i32 %v49)
  call void asm sideeffect "; use $0", "v"(i32 %v50)
  call void asm sideeffect "; use $0", "v"(i32 %v51)
  call void asm sideeffect "; use $0", "v"(i32 %v52)
  call void asm sideeffect "; use $0", "v"(i32 %v53)
  call void asm sideeffect "; use $0", "v"(i32 %v54)
  call void asm sideeffect "; use $0", "v"(i32 %v55)
  call void asm sideeffect "; use $0", "v"(i32 %v56)
  call void asm sideeffect "; use $0", "v"(i32 %v57)
  call void asm sideeffect "; use $0", "v"(i32 %v58)
  call void asm sideeffect "; use $0", "v"(i32 %v59)
  call void asm sideeffect "; use $0", "v"(i32 %v60)
  call void asm sideeffect "; use $0", "v"(i32 %v61)
  call void asm sideeffect "; use $0", "v"(i32 %v62)
  call void asm sideeffect "; use $0", "v"(i32 %v63)
  call void asm sideeffect "; use $0", "v"(i32 %v64)
  call void asm sideeffect "; use $0", "v"(i32 %v65)
  call void asm sideeffect "; use $0", "v"(i32 %v66)
  call void asm sideeffect "; use $0", "v"(i32 %v67)
  call void asm sideeffect "; use $0", "v"(i32 %v68)
  call void asm sideeffect "; use $0", "v"(i32 %v69)
  call void asm sideeffect "; use $0", "v"(i32 %v70)
  call void asm sideeffect "; use $0", "v"(i32 %v71)
  call void asm sideeffect "; use $0", "v"(i32 %v72)
  call void asm sideeffect "; use $0", "v"(i32 %v73)
  call void asm sideeffect "; use $0", "v"(i32 %v74)
  call void asm sideeffect "; use $0", "v"(i32 %v75)
  call void asm sideeffect "; use $0", "v"(i32 %v76)
  call void asm sideeffect "; use $0", "v"(i32 %v77)
  call void asm sideeffect "; use $0", "v"(i32 %v78)
  call void asm sideeffect "; use $0", "v"(i32 %v79)
  call void asm sideeffect "; use $0", "v"(i32 %v80)
  call void asm sideeffect "; use $0", "v"(i32 %v81)
  call void asm sideeffect "; use $0", "v"(i32 %v82)
  call void asm sideeffect "; use $0", "v"(i32 %v83)
  call void asm sideeffect "; use $0", "v"(i32 %v84)
  call void asm sideeffect "; use $0", "v"(i32 %v85)
  call void asm sideeffect "; use $0", "v"(i32 %v86)
  call void asm sideeffect "; use $0", "v"(i32 %v87)
  call void asm sideeffect "; use $0", "v"(i32 %v88)
  call void asm sideeffect "; use $0", "v"(i32 %v89)
  call void asm sideeffect "; use $0", "v"(i32 %v90)
  call void asm sideeffect "; use $0", "v"(i32 %v91)
  call void asm sideeffect "; use $0", "v"(i32 %v92)
  call void asm sideeffect "; use $0", "v"(i32 %v93)
  call void asm sideeffect "; use $0", "v"(i32 %v94)
  call void asm sideeffect "; use $0", "v"(i32 %v95)
  call void asm sideeffect "; use $0", "v"(i32 %v96)
  call void asm sideeffect "; use $0", "v"(i32 %v97)
  call void asm sideeffect "; use $0", "v"(i32 %v98)
  call void asm sideeffect "; use $0", "v"(i32 %v99)
  call void asm sideeffect "; use $0", "v"(i32 %v100)
  call void asm sideeffect "; use $0", "v"(i32 %v101)
  call void asm sideeffect "; use $0", "v"(i32 %v102)
  call void asm sideeffect "; use $0", "v"(i32 %v103)
  call void asm sideeffect "; use $0", "v"(i32 %v104)
  call void asm sideeffect "; use $0", "v"(i32 %v105)
  call void asm sideeffect "; use $0", "v"(i32 %v106)
  call void asm sideeffect "; use $0", "v"(i32 %v107)
  call void asm sideeffect "; use $0", "v"(i32 %v108)
  call void asm sideeffect "; use $0", "v"(i32 %v109)
  call void asm sideeffect "; use $0", "v"(i32 %v110)
  call void asm sideeffect "; use $0", "v"(i32 %v111)
  call void asm sideeffect "; use $0", "v"(i32 %v112)
  call void asm sideeffect "; use $0", "v"(i32 %v113)
  call void asm sideeffect "; use $0", "v"(i32 %v114)
  call void asm sideeffect "; use $0", "v"(i32 %v115)
  call void asm sideeffect "; use $0", "v"(i32 %v116)
  call void asm sideeffect "; use $0", "v"(i32 %v117)
  call void asm sideeffect "; use $0", "v"(i32 %v118)
  call void asm sideeffect "; use $0", "v"(i32 %v119)
  call void asm sideeffect "; use $0", "v"(i32 %v120)
  call void asm sideeffect "; use $0", "v"(i32 %v121)
  call void asm sideeffect "; use $0", "v"(i32 %v122)
  call void asm sideeffect "; use $0", "v"(i32 %v123)
  call void asm sideeffect "; use $0", "v"(i32 %v124)
  call void asm sideeffect "; use $0", "v"(i32 %v125)
  call void asm sideeffect "; use $0", "v"(i32 %v126)
  call void asm sideeffect "; use $0", "v"(i32 %v127)
  call void asm sideeffect "; use $0", "v"(i32 %v128)
  call void asm sideeffect "; use $0", "v"(i32 %v129)
  call void asm sideeffect "; use $0", "v"(i32 %v130)
  call void asm sideeffect "; use $0", "v"(i32 %v131)
  call void asm sideeffect "; use $0", "v"(i32 %v132)
  call void asm sideeffect "; use $0", "v"(i32 %v133)
  call void asm sideeffect "; use $0", "v"(i32 %v134)
  call void asm sideeffect "; use $0", "v"(i32 %v135)
  call void asm sideeffect "; use $0", "v"(i32 %v136)
  call void asm sideeffect "; use $0", "v"(i32 %v137)
  call void asm sideeffect "; use $0", "v"(i32 %v138)
  call void asm sideeffect "; use $0", "v"(i32 %v139)
  call void asm sideeffect "; use $0", "v"(i32 %v140)
  call void asm sideeffect "; use $0", "v"(i32 %v141)
  call void asm sideeffect "; use $0", "v"(i32 %v142)
  call void asm sideeffect "; use $0", "v"(i32 %v143)
  call void asm sideeffect "; use $0", "v"(i32 %v144)
  call void asm sideeffect "; use $0", "v"(i32 %v145)
  call void asm sideeffect "; use $0", "v"(i32 %v146)
  call void asm sideeffect "; use $0", "v"(i32 %v147)
  call void asm sideeffect "; use $0", "v"(i32 %v148)
  call void asm sideeffect "; use $0", "v"(i32 %v149)
  call void asm sideeffect "; use $0", "v"(i32 %v150)
  call void asm sideeffect "; use $0", "v"(i32 %v151)
  call void asm sideeffect "; use $0", "v"(i32 %v152)
  call void asm sideeffect "; use $0", "v"(i32 %v153)
  call void asm sideeffect "; use $0", "v"(i32 %v154)
  call void asm sideeffect "; use $0", "v"(i32 %v155)
  call void asm sideeffect "; use $0", "v"(i32 %v156)
  call void asm sideeffect "; use $0", "v"(i32 %v157)
  call void asm sideeffect "; use $0", "v"(i32 %v158)
  call void asm sideeffect "; use $0", "v"(i32 %v159)
  call void asm sideeffect "; use $0", "v"(i32 %v160)
  call void asm sideeffect "; use $0", "v"(i32 %v161)
  call void asm sideeffect "; use $0", "v"(i32 %v162)
  call void asm sideeffect "; use $0", "v"(i32 %v163)
  call void asm sideeffect "; use $0", "v"(i32 %v164)
  call void asm sideeffect "; use $0", "v"(i32 %v165)
  call void asm sideeffect "; use $0", "v"(i32 %v166)
  call void asm sideeffect "; use $0", "v"(i32 %v167)
  call void asm sideeffect "; use $0", "v"(i32 %v168)
  call void asm sideeffect "; use $0", "v"(i32 %v169)
  call void asm sideeffect "; use $0", "v"(i32 %v170)
  call void asm sideeffect "; use $0", "v"(i32 %v171)
  call void asm sideeffect "; use $0", "v"(i32 %v172)
  call void asm sideeffect "; use $0", "v"(i32 %v173)
  call void asm sideeffect "; use $0", "v"(i32 %v174)
  call void asm sideeffect "; use $0", "v"(i32 %v175)
  call void asm sideeffect "; use $0", "v"(i32 %v176)
  call void asm sideeffect "; use $0", "v"(i32 %v177)
  call void asm sideeffect "; use $0", "v"(i32 %v178)
  call void asm sideeffect "; use $0", "v"(i32 %v179)
  call void asm sideeffect "; use $0", "v"(i32 %v180)
  call void asm sideeffect "; use $0", "v"(i32 %v181)
  call void asm sideeffect "; use $0", "v"(i32 %v182)
  call void asm sideeffect "; use $0", "v"(i32 %v183)
  call void asm sideeffect "; use $0", "v"(i32 %v184)
  call void asm sideeffect "; use $0", "v"(i32 %v185)
  call void asm sideeffect "; use $0", "v"(i32 %v186)
  call void asm sideeffect "; use $0", "v"(i32 %v187)
  call void asm sideeffect "; use $0", "v"(i32 %v188)
  call void asm sideeffect "; use $0", "v"(i32 %v189)
  call void asm sideeffect "; use $0", "v"(i32 %v190)
  call void asm sideeffect "; use $0", "v"(i32 %v191)
  call void asm sideeffect "; use $0", "v"(i32 %v192)
  call void asm sideeffect "; use $0", "v"(i32 %v193)
  call void asm sideeffect "; use $0", "v"(i32 %v194)
  call void asm sideeffect "; use $0", "v"(i32 %v195)
  call void asm sideeffect "; use $0", "v"(i32 %v196)
  call void asm sideeffect "; use $0", "v"(i32 %v197)
  call void asm sideeffect "; use $0", "v"(i32 %v198)
  call void asm sideeffect "; use $0", "v"(i32 %v199)
  call void asm sideeffect "; use $0", "v"(i32 %v200)
  call void asm sideeffect "; use $0", "v"(i32 %v201)
  call void asm sideeffect "; use $0", "v"(i32 %v202)
  call void asm sideeffect "; use $0", "v"(i32 %v203)
  call void asm sideeffect "; use $0", "v"(i32 %v204)
  call void asm sideeffect "; use $0", "v"(i32 %v205)
  call void asm sideeffect "; use $0", "v"(i32 %v206)
  call void asm sideeffect "; use $0", "v"(i32 %v207)
  call void asm sideeffect "; use $0", "v"(i32 %v208)
  call void asm sideeffect "; use $0", "v"(i32 %v209)
  call void asm sideeffect "; use $0", "v"(i32 %v210)
  call void asm sideeffect "; use $0", "v"(i32 %v211)
  call void asm sideeffect "; use $0", "v"(i32 %v212)
  call void asm sideeffect "; use $0", "v"(i32 %v213)
  call void asm sideeffect "; use $0", "v"(i32 %v214)
  call void asm sideeffect "; use $0", "v"(i32 %v215)
  call void asm sideeffect "; use $0", "v"(i32 %v216)
  call void asm sideeffect "; use $0", "v"(i32 %v217)
  call void asm sideeffect "; use $0", "v"(i32 %v218)
  call void asm sideeffect "; use $0", "v"(i32 %v219)
  call void asm sideeffect "; use $0", "v"(i32 %v220)
  call void asm sideeffect "; use $0", "v"(i32 %v221)
  call void asm sideeffect "; use $0", "v"(i32 %v222)
  call void asm sideeffect "; use $0", "v"(i32 %v223)
  call void asm sideeffect "; use $0", "v"(i32 %v224)
  call void asm sideeffect "; use $0", "v"(i32 %v225)
  call void asm sideeffect "; use $0", "v"(i32 %v226)
  call void asm sideeffect "; use $0", "v"(i32 %v227)
  call void asm sideeffect "; use $0", "v"(i32 %v228)
  call void asm sideeffect "; use $0", "v"(i32 %v229)
  call void asm sideeffect "; use $0", "v"(i32 %v230)
  call void asm sideeffect "; use $0", "v"(i32 %v231)
  call void asm sideeffect "; use $0", "v"(i32 %v232)
  call void asm sideeffect "; use $0", "v"(i32 %v233)
  call void asm sideeffect "; use $0", "v"(i32 %v234)
  call void asm sideeffect "; use $0", "v"(i32 %v235)
  call void asm sideeffect "; use $0", "v"(i32 %v236)
  call void asm sideeffect "; use $0", "v"(i32 %v237)
  call void asm sideeffect "; use $0", "v"(i32 %v238)
  call void asm sideeffect "; use $0", "v"(i32 %v239)
  call void asm sideeffect "; use $0", "v"(i32 %v240)
  call void asm sideeffect "; use $0", "v"(i32 %v241)
  call void asm sideeffect "; use $0", "v"(i32 %v242)
  call void asm sideeffect "; use $0", "v"(i32 %v243)
  call void asm sideeffect "; use $0", "v"(i32 %v244)
  call void asm sideeffect "; use $0", "v"(i32 %v245)
  call void asm sideeffect "; use $0", "v"(i32 %v246)
  call void asm sideeffect "; use $0", "v"(i32 %v247)
  call void asm sideeffect "; use $0", "v"(i32 %v248)
  call void asm sideeffect "; use $0", "v"(i32 %v249)
  call void asm sideeffect "; use $0", "v"(i32 %v250)
  call void asm sideeffect "; use $0", "v"(i32 %v251)
  call void asm sideeffect "; use $0", "v"(i32 %v252)
  call void asm sideeffect "; use $0", "v"(i32 %v253)
  call void asm sideeffect "; use $0", "v"(i32 %v254)
  call void asm sideeffect "; use $0", "v"(i32 %v255)
  ret void
}

define internal void @use512vgprs() {
  %v0 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  %v1 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  %v2 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  %v3 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  %v4 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  %v5 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  %v6 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  %v7 = call <32 x i32> asm sideeffect "; def $0", "=v"()
  call void @use256vgprs()
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v0)
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v1)
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v2)
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v3)
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v4)
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v5)
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v6)
  call void asm sideeffect "; use $0", "v"(<32 x i32> %v7)
  ret void
}

define void @foo() #0 {
  ret void
}

attributes #0 = { noinline }

; GCN-LABEL: {{^}}k256_w8:
; GFX90A: NumVgprs: 32
; GFX90A: NumAgprs: 32
; GFX90A: TotalNumVgprs: 64
define amdgpu_kernel void @k256_w8() #2568 {
  call void @foo()
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}k256_w8_no_agprs:
; GFX90A: NumVgprs: 64
; GFX90A: NumAgprs: 0
; GFX90A: TotalNumVgprs: 64
define amdgpu_kernel void @k256_w8_no_agprs() #2568 {
  call void @use256vgprs()
  ret void
}

attributes #2568 = { nounwind "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="8" }

; GCN-LABEL: {{^}}k256_w4:
; GFX90A: NumVgprs: 64
; GFX90A: NumAgprs: 64
; GFX90A: TotalNumVgprs: 128
define amdgpu_kernel void @k256_w4() #2564 {
  call void @foo()
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}k256_w4_no_agprs:
; GFX90A: NumVgprs: 128
; GFX90A: NumAgprs: 0
; GFX90A: TotalNumVgprs: 128
define amdgpu_kernel void @k256_w4_no_agprs() #2564 {
  call void @use256vgprs()
  ret void
}

attributes #2564 = { nounwind "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="4" }

; GCN-LABEL: {{^}}k256_w2:
; GFX90A: NumVgprs: 128
; GFX90A: NumAgprs: 128
; GFX90A: TotalNumVgprs: 256
define amdgpu_kernel void @k256_w2() #2562 {
  call void @foo()
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}k256_w2_no_agprs:
; GFX90A: NumVgprs: 256
; GFX90A: NumAgprs: 0
; GFX90A: TotalNumVgprs: 256
define amdgpu_kernel void @k256_w2_no_agprs() #2562 {
  call void @use256vgprs()
  ret void
}

attributes #2562 = { nounwind "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="2" }

; GCN-LABEL: {{^}}k256_w1:
; GFX90A: NumVgprs: 256
; GFX90A: NumAgprs: 256
; GFX90A: TotalNumVgprs: 512
define amdgpu_kernel void @k256_w1() #2561 {
  call void @foo()
  call void @use512vgprs()
  ret void
}

; GCN-LABEL: {{^}}k256_w1_no_agprs:
; GFX90A: NumVgprs: 256
; GFX90A: NumAgprs: 256
; GFX90A: TotalNumVgprs: 512
define amdgpu_kernel void @k256_w1_no_agprs() #2561 {
  call void @use512vgprs()
  ret void
}

attributes #2561 = { nounwind "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" }

; GCN-LABEL: {{^}}k512_no_agprs:
; GFX90A: NumVgprs: 256
; GFX90A: NumAgprs: 0
; GFX90A: TotalNumVgprs: 256
define amdgpu_kernel void @k512_no_agprs() #512 {
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}k512_call:
; GFX90A: NumVgprs: 128
; GFX90A: NumAgprs: 128
; GFX90A: TotalNumVgprs: 256
define amdgpu_kernel void @k512_call() #512 {
  call void @foo()
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}k512_virtual_agpr:
; GFX90A: NumVgprs: 128
; GFX90A: NumAgprs: 128
; GFX90A: TotalNumVgprs: 256
define amdgpu_kernel void @k512_virtual_agpr() #512 {
  %a0 = call i32 asm sideeffect "; def $0", "=a"()
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}k512_physical_agpr:
; GFX90A: NumVgprs: 128
; GFX90A: NumAgprs: 128
; GFX90A: TotalNumVgprs: 256
define amdgpu_kernel void @k512_physical_agpr() #512 {
  call void asm sideeffect "", "~{a8}" ()
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}f512:
; GFX90A: NumVgprs: 12{{[0-9]}}
; GFX90A: NumAgprs: {{[1-9]}}
define void @f512() #512 {
  call void @use256vgprs()
  ret void
}

attributes #512 = { nounwind "amdgpu-flat-work-group-size"="512,512" }

; GCN-LABEL: {{^}}k1024:
; GFX90A: NumVgprs: 128
; GFX90A: NumAgprs: 0
; GFX90A: TotalNumVgprs: 128
define amdgpu_kernel void @k1024() #1024 {
  call void @use256vgprs()
  ret void
}

; GCN-LABEL: {{^}}k1024_call:
; GFX90A: NumVgprs: 64
; GFX90A: NumAgprs: 64
; GFX90A: TotalNumVgprs: 128
define amdgpu_kernel void @k1024_call() #1024 {
  call void @foo()
  call void @use256vgprs()
  ret void
}

attributes #1024 = { nounwind "amdgpu-flat-work-group-size"="1024,1024" }
