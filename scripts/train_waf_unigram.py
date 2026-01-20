#!/usr/bin/env python3
from __future__ import annotations

from scripts import train_tokenizer as _impl

parse_args = _impl.parse_args
run_with_auto_retry = _impl.run_with_auto_retry
train_once = _impl.train_once
main = _impl.main


if __name__ == "__main__":
    main()
