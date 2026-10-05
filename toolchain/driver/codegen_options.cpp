// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/codegen_options.h"

#include <string>

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace Carbon {

auto CodegenOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringOption(
      {
          .name = "target",
          .help = R"""(
Select a target platform. Uses the LLVM target syntax. Also known as a "triple"
for historical reasons.

This corresponds to the `target` flag to Clang and accepts the same strings
documented there:
https://clang.llvm.org/docs/CrossCompilation.html#target-triple
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(host);
        arg_b.Set(&target);
      });

  b.AddStringOption(
      {
          .name = "target-cpu",
          .value_name = "CPU",
          .help = R"""(
Select a target CPU or architecture level.

Accepts CPU names (such as `znver4`, `apple-m1`, `spacemit-x60`, or `native`) as
well as architecture strings or profiles (such as `x86-64-v3`, `armv9-a`,
`rv64gc`, or `rva22u64`). This corresponds to Clang's `-march` and `-mcpu`
flags, selecting the appropriate underlying flag for the target architecture.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&target_cpu); });

  b.AddStringOption(
      {
          .name = "target-cpu-tune",
          .value_name = "CPU",
          .help = R"""(
Select a target CPU to tune code generation for without enabling new
instruction set extensions.

Accepts CPU names (such as `znver4`, `apple-m1`, `spacemit-x60`, `generic`, or
`native`). This corresponds to Clang's `-mtune` flag.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&target_cpu_tune); });

  b.AddStringOption(
      {
          .name = "target-cpu-features",
          .value_name = "FEATURES",
          .help = R"""(
Select target CPU features to enable or disable.

Accepts a comma-separated list of LLVM target feature names, optionally prefixed
with `+` to enable or `-` to disable (such as `+avx2,-fma` or `sve,crc`). Bare
feature names without a prefix are enabled.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&target_cpu_features); });
}

static auto AppendTargetCpuClangArgs(llvm::StringRef target,
                                     llvm::StringRef target_cpu,
                                     llvm::SmallVectorImpl<std::string>& args)
    -> void {
  if (target_cpu.empty()) {
    return;
  }

  llvm::Triple triple(target);
  if (triple.isX86()) {
    // On x86, Clang uses `-march=` for both microarchitecture levels (such as
    // `x86-64-v3`) and specific CPUs (such as `znver4` or `native`), and does
    // not support `-mcpu=`.
    args.push_back(llvm::formatv("-march={0}", target_cpu).str());
    return;
  }

  if (triple.isAArch64()) {
    // On AArch64, Clang's driver checks `-mcpu=` first for the target CPU name,
    // but checks `-march=` first for target features. Emit both `-march=` and
    // `-mcpu=` so `--target-cpu` overrides both in earlier Clang arguments.
    auto [base, mods] = target_cpu.split('+');
    if (llvm::AArch64::parseArch(base) != nullptr || base.starts_with("armv")) {
      args.push_back(llvm::formatv("-march={0}", target_cpu).str());
      args.push_back("-mcpu=generic");
      return;
    }
    if (base.equals_insensitive("native")) {
      args.push_back(llvm::formatv("-march={0}", target_cpu).str());
      args.push_back(llvm::formatv("-mcpu={0}", target_cpu).str());
      return;
    }
    if (std::optional<llvm::AArch64::CpuInfo> cpu_info =
            llvm::AArch64::parseCpu(base.lower())) {
      const llvm::AArch64::ArchInfo& arch_info =
          llvm::AArch64::ArchInfos[cpu_info->ArchIdx];
      llvm::AArch64::ExtensionSet cpu_exts;
      cpu_exts.addCPUDefaults(*cpu_info);
      std::string march =
          llvm::formatv("-march={0}", llvm::AArch64::StrTab[arch_info.Name])
              .str();
      for (const auto& ext : llvm::AArch64::Extensions) {
        if (!cpu_exts.Enabled.test(ext.ID) ||
            arch_info.DefaultExts.test(ext.ID)) {
          continue;
        }
        llvm::StringRef ext_name = llvm::AArch64::StrTab[ext.UserVisibleName];
        if (!ext_name.empty()) {
          march += "+";
          march += ext_name;
        }
      }
      if (!mods.empty()) {
        march += "+";
        march += mods;
        llvm::SmallVector<llvm::StringRef> mod_list;
        mods.split(mod_list, '+', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        for (llvm::StringRef mod : mod_list) {
          cpu_exts.parseModifier(mod);
        }
      }
      args.push_back(std::move(march));
      args.push_back(llvm::formatv("-mcpu={0}", target_cpu).str());
      for (const auto& ext : llvm::AArch64::Extensions) {
        if (!cpu_exts.Enabled.test(ext.ID) ||
            arch_info.DefaultExts.test(ext.ID) ||
            !llvm::AArch64::StrTab[ext.UserVisibleName].empty()) {
          continue;
        }
        llvm::StringRef pos_feature =
            llvm::AArch64::StrTab[ext.PosTargetFeature];
        if (!pos_feature.empty()) {
          args.append(
              {"-Xclang", "-target-feature", "-Xclang", pos_feature.str()});
        }
      }
      return;
    }
    args.push_back("-march=armv8-a");
    args.push_back(llvm::formatv("-mcpu={0}", target_cpu).str());
    return;
  }

  if (triple.isRISCV()) {
    // On RISC-V, Clang's driver checks `-mcpu=` first for the target CPU name,
    // and checks `-march=` first for target features unless `-march=unset` is
    // passed (which resets `-march=` so features are derived from `-mcpu=`).
    // Emit both `-march=` and `-mcpu=` so `--target-cpu` overrides both.
    auto isa_info = llvm::RISCVISAInfo::parseArchString(
        target_cpu, /*EnableExperimentalExtension=*/true);
    bool is_arch = static_cast<bool>(isa_info);
    if (!isa_info) {
      llvm::consumeError(isa_info.takeError());
      is_arch =
          !llvm::RISCV::parseCPU(target_cpu, triple.isRISCV64()) &&
          (target_cpu.starts_with("rv32") || target_cpu.starts_with("rv64") ||
           target_cpu.starts_with("rva") || target_cpu.starts_with("rvb") ||
           target_cpu.starts_with("rvi") || target_cpu.starts_with("rvm"));
    }
    if (is_arch) {
      args.push_back(llvm::formatv("-march={0}", target_cpu).str());
      args.push_back(triple.isRISCV64() ? "-mcpu=generic-rv64"
                                        : "-mcpu=generic-rv32");
    } else {
      args.push_back("-march=unset");
      args.push_back(llvm::formatv("-mcpu={0}", target_cpu).str());
    }
    return;
  }

  args.push_back(llvm::formatv("-mcpu={0}", target_cpu).str());
}

auto CodegenOptions::AppendClangArgs(
    llvm::SmallVectorImpl<std::string>& args) const -> void {
  AppendTargetCpuClangArgs(target, target_cpu, args);
  if (!target_cpu_tune.empty()) {
    args.push_back(llvm::formatv("-mtune={0}", target_cpu_tune).str());
  }
  if (!target_cpu_features.empty()) {
    llvm::SmallVector<llvm::StringRef> features;
    target_cpu_features.split(features, ',', /*MaxSplit=*/-1,
                              /*KeepEmpty=*/false);
    for (llvm::StringRef feature : features) {
      args.push_back("-Xclang");
      args.push_back("-target-feature");
      args.push_back("-Xclang");
      if (feature.size() > 1 &&
          (feature.starts_with('+') || feature.starts_with('-'))) {
        args.push_back(feature.str());
      } else {
        args.push_back(llvm::formatv("+{0}", feature).str());
      }
    }
  }
}

auto CodegenOptions::GetClangArgs() const -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> args;
  AppendClangArgs(args);
  return args;
}

}  // namespace Carbon
