from matchpoint.core import MatchPoint

try:
    from ._version import version as __version__
except ImportError:
    try:
        import setuptools_scm
        __version__ = setuptools_scm.get_version(version_scheme="post-release")
    except (LookupError, ImportError):
        __version__ = '0.0.0'