Lessons

- the `__init__.py` file treats teh folder as a package where we can import.
- `ci.yml` is a GitHUb Actions workflow file that defines a CI (Continuous Integration) pipeline that runs automated checks on your code every time you push or open a PR.
- `PYTHONPATH=. python3 scripts/test_logger.py` have to use the `PYTHNOPATH=.` for development
- test_logger.py goes in `scripts/` folder
- CI part will do later with the `.yml` file.