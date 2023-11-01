// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleImportIntroducer(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleImportIntroducer");
}

auto HandlePackageIntroducer(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandlePackageIntroducer");
}

auto HandleLibrary(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleLibrary");
}

auto HandlePackageApi(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandlePackageApi");
}

auto HandlePackageImpl(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandlePackageImpl");
}

auto HandleImportDirective(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleImportDirective");
}

auto HandlePackageDirective(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandlePackageDirective");
}

}  // namespace Carbon::Check
