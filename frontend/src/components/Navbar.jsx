import React, { useState } from 'react'

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)

  const toggleMenu = () => {
    setIsOpen(!isOpen)
  }

  return (
    <nav className="bg-gradient-to-r from-blue-900 to-green-800 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center mr-3">
                <span className="text-blue-900 font-bold text-lg">🌍</span>
              </div>
              <h1 className="text-white text-xl font-bold">Planetary</h1>
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <a href="#dashboard" className="text-white hover:bg-blue-700 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                Dashboard
              </a>
              <a href="#about" className="text-white hover:bg-blue-700 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                About
              </a>
              <a href="#contact" className="text-white hover:bg-blue-700 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                Contact
              </a>
              <a href="#docs" className="text-white hover:bg-blue-700 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                Docs
              </a>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={toggleMenu}
              className="text-white hover:text-gray-300 focus:outline-none focus:text-gray-300"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {isOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-blue-800 rounded-lg mt-2">
              <a href="#dashboard" className="text-white hover:bg-blue-700 block px-3 py-2 rounded-md text-base font-medium">
                Dashboard
              </a>
              <a href="#about" className="text-white hover:bg-blue-700 block px-3 py-2 rounded-md text-base font-medium">
                About
              </a>
              <a href="#contact" className="text-white hover:bg-blue-700 block px-3 py-2 rounded-md text-base font-medium">
                Contact
              </a>
              <a href="#docs" className="text-white hover:bg-blue-700 block px-3 py-2 rounded-md text-base font-medium">
                Docs
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

export default Navbar
