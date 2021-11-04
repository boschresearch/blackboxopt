"""Generate "virtual" doc files for the api references.

The files are only generated during build time and not actually written on disk.
To write them on disk (for e.g. debugging) execute this script directely.
"""
from pathlib import Path

import mkdocs_gen_files

module_name = mkdocs_gen_files.config["plugins"]["gen-files"].config["module"]
excludes = mkdocs_gen_files.config["plugins"]["gen-files"].config["exclude"]

src_root = Path(module_name)
for path in src_root.glob("**/*.py"):
    doc_path = Path("reference", path.relative_to(src_root)).with_suffix(".md")

    if any([exclude in str(doc_path) for exclude in excludes]):
        continue

    with mkdocs_gen_files.open(doc_path, "w") as f:
        ident = ".".join(path.with_suffix("").parts)
        print("::: " + ident, file=f)
