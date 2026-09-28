# Gemini & AI Assistant Guide for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document provides high-density technical context for AI assistants
contributing to the Carbon Language project.

## General instructions

-   **Communication**: Be concise, professional, and technical. Use GitHub-style
    markdown.
-   **Verification**: Always run relevant tests.

## Bazel usage

> [!IMPORTANT] Always use `bazelisk` instead of `bazel` for all commands in the
> Carbon project. Refer to the
> [Bazel usage skill](/.agents/skills/bazel/SKILL.md) for detailed instructions.

## Version control

> [!IMPORTANT] Never rewrite the history of a change that has been submitted as
> a pull request. Reviewers track a PR by its commits, and rewriting them
> discards their in-progress review. Ask before rewriting history in any case.
> Refer to the [Jujutsu (jj) usage skill](/.agents/skills/jj/SKILL.md) for
> details.
