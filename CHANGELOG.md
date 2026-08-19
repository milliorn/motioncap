# Changelog

## [0.10.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.9.0...motioncap-v0.10.0) (2026-08-18)


### Features

* **opencv_utils:** add bgr_vec_to_mat function and related error handling test ([#34](https://github.com/milliorn/motioncap/issues/34)) ([7732d0b](https://github.com/milliorn/motioncap/commit/7732d0b111d2162d122077de7a0ff035ef9b76c8))

## [0.9.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.8.0...motioncap-v0.9.0) (2026-08-16)


### Features

* **config:** refactor argument parsing and improve coverage documentation ([#29](https://github.com/milliorn/motioncap/issues/29)) ([c2664ca](https://github.com/milliorn/motioncap/commit/c2664ca88975c40609de951415544f47be99cd61))

## [0.8.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.7.0...motioncap-v0.8.0) (2026-08-16)


### Features

* **recorder:** gate new recordings on pre-buffer readiness after camera reconnect ([#27](https://github.com/milliorn/motioncap/issues/27)) ([85ad86c](https://github.com/milliorn/motioncap/commit/85ad86c5360a4eb40361ef42948da3427e655938))

## [0.7.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.6.0...motioncap-v0.7.0) (2026-08-16)


### Features

* implement pre-buffer readiness check for recording start after camera reconnect ([01e4c85](https://github.com/milliorn/motioncap/commit/01e4c85b87d6a5c992fbf05e761c41391035c721))
* optimize detection polling by skipping inference during pre-buffer wait ([25471a3](https://github.com/milliorn/motioncap/commit/25471a30e687e342a64d1ee826f53e91c38db279))

## [0.6.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.5.1...motioncap-v0.6.0) (2026-08-15)


### Features

* add PR title linting and document audio sample format support ([#23](https://github.com/milliorn/motioncap/issues/23)) ([9b7dbeb](https://github.com/milliorn/motioncap/commit/9b7dbeb42d4b552e2eb4cc391d2b85b1443025da))

## [0.5.1](https://github.com/milliorn/motioncap/compare/motioncap-v0.5.0...motioncap-v0.5.1) (2026-08-13)


### Bug Fixes

* **detection:** trigger release for repeat-sighting confirmation gate ([#18](https://github.com/milliorn/motioncap/issues/18)) ([cb0c3f0](https://github.com/milliorn/motioncap/commit/cb0c3f022d3e5b23ee127387e8d9e02d48d695a2))

## [0.5.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.4.0...motioncap-v0.5.0) (2026-08-13)


### Features

* **camera:** reconnect stalled camera stream automatically ([#13](https://github.com/milliorn/motioncap/issues/13)) ([7724f10](https://github.com/milliorn/motioncap/commit/7724f1004da7ef9702012a69009e79f4bb575945))

## [0.4.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.3.0...motioncap-v0.4.0) (2026-08-12)


### Features

* **ci:** warm OpenCV/cargo cache on push to main ([#15](https://github.com/milliorn/motioncap/issues/15)) ([24f0b54](https://github.com/milliorn/motioncap/commit/24f0b54b7a18e7946c6a3cf937f66a40c61362e3))

## [0.3.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.2.0...motioncap-v0.3.0) (2026-08-10)


### Features

* **recorder:** track motion-gate trips and improve logging ([#11](https://github.com/milliorn/motioncap/issues/11)) ([30d5747](https://github.com/milliorn/motioncap/commit/30d57473ab9f11a23d1a13eab28c9a03907ba1fe))

## [0.2.0](https://github.com/milliorn/motioncap/compare/motioncap-v0.1.0...motioncap-v0.2.0) (2026-08-09)


### Features

* initialize motioncap project with basic setup and dependencies ([f607243](https://github.com/milliorn/motioncap/commit/f607243475df7162854996251259bd4232da9ee2))
* **main:** implement separate detection and preview loops for performance ([#1](https://github.com/milliorn/motioncap/issues/1)) ([abd80ac](https://github.com/milliorn/motioncap/commit/abd80ace34d1796b81ed2cec89d03d962184a8a9))
