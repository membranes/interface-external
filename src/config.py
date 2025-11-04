import os

class Config:

    def __init__(self):
        """
        Constructor
        """

        # Examples
        self.examples = [
            ['The English writer and the Afghani soldier.'],
            ['It was written by members of the United Nation.'],
            [('There were more than a hundred wolves in the Tiger Basin.  It is a dangerous place '
              'after 9 p.m., especially near Lake Victoria.')]
        ]

        self.model_ = os.path.join(os.getcwd(), 'src', 'data', 'model')
        self.config_ = os.path.join(self.model_, 'config.json')
