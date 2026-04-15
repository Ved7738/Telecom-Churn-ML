# This file is intentionally empty.
#
# WHY IT EXISTS:
#   Python needs this file to treat the "src" folder as a PACKAGE.
#   A package is just a folder of related Python files that can be imported together.
#
#   Without this file, you cannot write:
#       from src.preprocessing import DataPreprocessor
#
#   With this file, Python knows "src" is a package and allows the import above.
#
# ANALOGY:
#   Think of it like a Table of Contents page in a book.
#   The book (folder) needs a ToC page (__init__.py) to be recognised as a proper book.
