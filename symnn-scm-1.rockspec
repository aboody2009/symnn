package = 'symnn'
version = 'scm-1'

source = {
   url = 'git://github.com/benglard/symnn',
}

description = {
   summary = 'Neural networks for symtorch',
   detailed = 'Neural networks for symtorch',
   homepage = 'https://github.com/benglard/symnn'
}

dependencies = {
   'torch >= 7.0',
}

build = {
   type = 'builtin',
   modules = {
      ['symnn.init'] = 'symnn/init.lua',
      ['symnn.container'] = 'symnn/container.lua',
      ['symnn.conv'] = 'symnn/conv.lua',
      ['symnn.criterions'] = 'symnn/criterions.lua',
      ['symnn.linear'] = 'symnn/linear.lua',
      ['symnn.nonlinearities'] = 'symnn/nonlinearities.lua',
      ['symnn.recurrent'] = 'symnn/recurrent.lua',
   }
}