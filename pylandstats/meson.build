
python_sources = [
  '__init__.py',
  'landscape.py',
  'multilandscape.py',
  'settings.py',
  'spatiotemporal.py',
  'zonal.py',
]

py.install_sources(
  python_sources,
  subdir: 'pylandstats'
)

run_command(['transonic', '--meson', '--backend', backend, 'landscape.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
