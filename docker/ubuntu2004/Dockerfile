FROM carbon-base as runner

RUN git clone https://github.com/carbon-language/carbon-lang

WORKDIR /carbon-lang

RUN pre-commit install

CMD ["bazel", "run", "//explorer", "--", "./explorer/testdata/print/format_only.carbon"]