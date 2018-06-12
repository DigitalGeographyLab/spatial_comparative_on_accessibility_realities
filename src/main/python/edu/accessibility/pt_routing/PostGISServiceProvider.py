from sqlalchemy import create_engine

from src.main.python.edu.accessibility.util.utilitaries import getConfigurationProperties


class PostGISServiceProvider(object):
    # engine = create_engine('postgresql://<yourUserName>:postgres@localhost:5432/postgres', echo=False)
    # Session = sessionmaker(bind=engine)
    # session = Session()
    # meta = MetaData(engine, schema='cldmatchup')
    def __init__(self):
        self.__engine = None

    def getEngine(self):
        if not self.__engine:
            config = getConfigurationProperties(section="DATABASE_CONFIG")
            # engine = create_engine('postgresql://<yourUserName>:postgres@localhost:5432/postgres', echo=False)
            self.__engine = create_engine(
                'postgresql://%s:%s@%s/%s' % (
                    config["user"], config["password"], config["host"], config["database_name"]),
                echo=False
            )

        return self.__engine