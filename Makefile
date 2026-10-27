.PHONY: vendor deb test

DEB_HOST_ARCH ?= $(shell dpkg-architecture -qDEB_HOST_ARCH)

# Only preparation fetches upstream release files; `deb` never uses the network.
# NODE_EXPORTER_ARCHIVE permits checksum-verified preparation from a local tarball.
vendor:
	python3 debian/prepare-node-exporter.py --architecture "$(DEB_HOST_ARCH)" $(if $(NODE_EXPORTER_ARCHIVE),--archive "$(NODE_EXPORTER_ARCHIVE)")

# Binary-only builds need no orig tarball, including with 3.0 (quilt).
# dpkg-buildpackage writes the .deb, .buildinfo and .changes to ../.
deb:
	dpkg-buildpackage -us -uc -b

test:
	python3 -m unittest discover -v
