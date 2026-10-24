# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Geometry: NACA 4-digit contours (closed trailing edge, cosine spacing, counter-clockwise
  from the trailing edge) and panel construction (midpoints, lengths, tangents, outward
  normals), with tests for closure, thickness, symmetry and normal direction.
- Project spine: packaging, `foilnet` CLI stub, TOML config loader, CI on
  Ubuntu and Windows with Python 3.11 and 3.12.
