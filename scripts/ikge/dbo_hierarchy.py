"""
DBPedia Ontology (DBO) type hierarchy — manually extracted from:
  http://mappings.dbpedia.org/server/ontology/classes/
  https://dbpedia.org/ontology/ (2016-10 version)

Maps each leaf/intermediate type to its full chain of ancestor types
(excluding owl:Thing which we treat separately via dbo:Thing).

Usage:
    from dbo_hierarchy import DBO_ANCESTORS
    # Get all ancestor types for a given type
    for t in DBO_ANCESTORS.get('dbo:MusicalArtist', []):
        print(t)
"""

# Parent-of mapping: child -> immediate parent.
# Covers only types present in DBPedia50k+ (from types_output.txt).
_DBO_PARENTS: dict[str, str] = {
    # ── Agent branch ──────────────────────────────────────────────────
    "dbo:Person":                "dbo:Agent",
    "dbo:Organisation":          "dbo:Agent",

    # Person subtypes
    "dbo:Athlete":               "dbo:Person",
    "dbo:Artist":                "dbo:Person",
    "dbo:Politician":            "dbo:Person",
    "dbo:MilitaryPerson":        "dbo:Person",
    "dbo:SportsTeamMember":      "dbo:Person",
    "dbo:Scientist":             "dbo:Person",
    "dbo:OfficeHolder":          "dbo:Person",
    "dbo:Royalty":               "dbo:Person",
    "dbo:Philosopher":           "dbo:Person",
    "dbo:Cleric":                "dbo:Person",
    "dbo:Judge":                 "dbo:Person",
    "dbo:Model":                 "dbo:Person",
    "dbo:PlayboyPlaymate":       "dbo:Person",
    "dbo:BeautyQueen":           "dbo:Person",
    "dbo:FashionDesigner":       "dbo:Person",
    "dbo:Chef":                  "dbo:Person",
    "dbo:Comedian":              "dbo:Artist",
    "dbo:ComedyGroup":           "dbo:Person",
    "dbo:RadioHost":             "dbo:Person",
    "dbo:Photographer":          "dbo:Artist",

    # Athlete subtypes
    "dbo:SoccerPlayer":          "dbo:Athlete",
    "dbo:BasketballPlayer":      "dbo:Athlete",
    "dbo:BaseballPlayer":        "dbo:Athlete",
    "dbo:IceHockeyPlayer":       "dbo:Athlete",
    "dbo:TennisPlayer":          "dbo:Athlete",
    "dbo:Swimmer":               "dbo:Athlete",
    "dbo:GolfPlayer":            "dbo:Athlete",
    "dbo:Wrestler":              "dbo:Athlete",
    "dbo:MartialArtist":         "dbo:Athlete",
    "dbo:Jockey":                "dbo:Athlete",
    "dbo:FormulaOneRacer":       "dbo:Athlete",
    "dbo:RacingDriver":          "dbo:Athlete",
    "dbo:MotorcycleRider":       "dbo:RacingDriver",
    "dbo:Cricketer":             "dbo:Athlete",
    "dbo:AmericanFootballPlayer":"dbo:Athlete",
    "dbo:GridironFootballPlayer":"dbo:Athlete",
    "dbo:FigureSkater":          "dbo:Athlete",
    "dbo:CollegeCoach":          "dbo:Person",
    "dbo:SoccerManager":         "dbo:Person",
    "dbo:HorseTrainer":          "dbo:Person",
    "dbo:PokerPlayer":           "dbo:Person",
    "dbo:HorseRace":             "dbo:SportsEvent",

    # Artist subtypes
    "dbo:MusicalArtist":         "dbo:Artist",
    "dbo:Band":                  "dbo:MusicalArtist",
    "dbo:Writer":                "dbo:Artist",
    "dbo:ScreenWriter":          "dbo:Writer",
    "dbo:ComicsCreator":         "dbo:Artist",
    "dbo:Actor":                 "dbo:Artist",
    "dbo:VoiceActor":            "dbo:Actor",
    "dbo:Guitarist":             "dbo:MusicalArtist",
    "dbo:Architect":             "dbo:Artist",

    # Politician subtypes
    "dbo:President":             "dbo:Politician",
    "dbo:PrimeMinister":         "dbo:Politician",
    "dbo:Governor":              "dbo:Politician",
    "dbo:Senator":               "dbo:Politician",
    "dbo:Congressman":           "dbo:Politician",
    "dbo:MemberOfParliament":    "dbo:Politician",
    "dbo:Chancellor":            "dbo:Politician",
    "dbo:Monarch":               "dbo:Royalty",

    # Scientist subtypes
    "dbo:Economist":             "dbo:Scientist",
    "dbo:Engineer":              "dbo:Scientist",
    "dbo:Entomologist":          "dbo:Scientist",

    # Cleric subtypes
    "dbo:Pope":                  "dbo:Cleric",
    "dbo:Cardinal":              "dbo:Cleric",
    "dbo:ChristianBishop":       "dbo:Cleric",
    "dbo:Saint":                 "dbo:Cleric",
    "dbo:Religious":             "dbo:Cleric",

    # Organisation subtypes
    "dbo:SportsTeam":            "dbo:Organisation",
    "dbo:Company":               "dbo:Organisation",
    "dbo:EducationalInstitution":"dbo:Organisation",
    "dbo:GovernmentAgency":      "dbo:Organisation",
    "dbo:PoliticalParty":        "dbo:Organisation",
    "dbo:TradeUnion":            "dbo:Organisation",
    "dbo:Non-ProfitOrganisation":"dbo:Organisation",
    "dbo:MilitaryUnit":          "dbo:Organisation",
    "dbo:Legislature":           "dbo:Organisation",
    "dbo:Diocese":               "dbo:Organisation",
    "dbo:BroadcastNetwork":      "dbo:Organisation",
    "dbo:RadioStation":          "dbo:BroadcastNetwork",
    "dbo:TelevisionStation":     "dbo:BroadcastNetwork",
    "dbo:Publisher":             "dbo:Company",
    "dbo:RecordLabel":           "dbo:Company",
    "dbo:Bank":                  "dbo:Company",
    "dbo:Airline":               "dbo:Company",
    "dbo:BusCompany":            "dbo:Company",
    "dbo:PublicTransitSystem":   "dbo:Organisation",

    # SportsTeam subtypes
    "dbo:SoccerClub":            "dbo:SportsTeam",
    "dbo:HockeyTeam":            "dbo:SportsTeam",
    "dbo:BasketballTeam":        "dbo:SportsTeam",
    "dbo:BaseballTeam":          "dbo:SportsTeam",
    "dbo:AmericanFootballTeam":  "dbo:SportsTeam",
    "dbo:CanadianFootballTeam":  "dbo:SportsTeam",
    "dbo:RugbyClub":             "dbo:SportsTeam",
    "dbo:FormulaOneTeam":        "dbo:SportsTeam",
    "dbo:HandballTeam":          "dbo:SportsTeam",

    # SportsLeague subtypes
    "dbo:SportsLeague":          "dbo:Organisation",
    "dbo:SoccerLeague":          "dbo:SportsLeague",
    "dbo:BasketballLeague":      "dbo:SportsLeague",
    "dbo:IceHockeyLeague":       "dbo:SportsLeague",
    "dbo:AmericanFootballLeague":"dbo:SportsLeague",
    "dbo:TennisLeague":          "dbo:SportsLeague",
    "dbo:BaseballLeague":        "dbo:SportsLeague",
    "dbo:GolfLeague":            "dbo:SportsLeague",
    "dbo:RugbyLeague":           "dbo:SportsLeague",

    # EducationalInstitution subtypes
    "dbo:University":            "dbo:EducationalInstitution",
    "dbo:School":                "dbo:EducationalInstitution",
    "dbo:College":               "dbo:EducationalInstitution",

    # ── Place branch ──────────────────────────────────────────────────
    "dbo:Place":                 "dbo:Thing",
    "dbo:Location":              "dbo:Place",
    "dbo:PopulatedPlace":        "dbo:Place",
    "dbo:BodyOfWater":           "dbo:Place",
    "dbo:NaturalPlace":          "dbo:Place",
    "dbo:ArchitecturalStructure":"dbo:Place",
    "dbo:Infrastructure":        "dbo:ArchitecturalStructure",
    "dbo:ProtectedArea":         "dbo:Place",
    "dbo:WorldHeritageSite":     "dbo:ProtectedArea",

    # PopulatedPlace subtypes
    "dbo:Settlement":            "dbo:PopulatedPlace",
    "dbo:City":                  "dbo:Settlement",
    "dbo:Town":                  "dbo:Settlement",
    "dbo:Village":               "dbo:Settlement",
    "dbo:AdministrativeRegion": "dbo:PopulatedPlace",
    "dbo:Country":               "dbo:AdministrativeRegion",
    "dbo:Island":                "dbo:PopulatedPlace",
    "dbo:Continent":             "dbo:PopulatedPlace",

    # BodyOfWater subtypes
    "dbo:River":                 "dbo:BodyOfWater",
    "dbo:Lake":                  "dbo:BodyOfWater",
    "dbo:Sea":                   "dbo:BodyOfWater",

    # NaturalPlace subtypes
    "dbo:Mountain":              "dbo:NaturalPlace",
    "dbo:MountainRange":         "dbo:NaturalPlace",
    "dbo:Volcano":               "dbo:Mountain",

    # ArchitecturalStructure subtypes
    "dbo:Building":              "dbo:ArchitecturalStructure",
    "dbo:HistoricBuilding":      "dbo:Building",
    "dbo:HistoricPlace":         "dbo:Place",
    "dbo:ReligiousBuilding":     "dbo:Building",
    "dbo:Stadium":               "dbo:ArchitecturalStructure",
    "dbo:Venue":                 "dbo:ArchitecturalStructure",
    "dbo:Theatre":               "dbo:Venue",
    "dbo:Museum":                "dbo:Building",
    "dbo:MilitaryStructure":     "dbo:Building",
    "dbo:ConcentrationCamp":     "dbo:MilitaryStructure",

    # Infrastructure subtypes
    "dbo:Road":                  "dbo:Infrastructure",
    "dbo:RailwayLine":           "dbo:Infrastructure",
    "dbo:RailwayStation":        "dbo:Infrastructure",
    "dbo:Airport":               "dbo:Infrastructure",

    # Misc Place
    "dbo:Park":                  "dbo:Place",
    "dbo:WineRegion":            "dbo:AdministrativeRegion",
    "dbo:Racecourse":            "dbo:Venue",

    # ── Species/Taxon branch ──────────────────────────────────────────
    "dbo:Species":               "dbo:Eukaryote",
    "dbo:Animal":                "dbo:Eukaryote",
    "dbo:Plant":                 "dbo:Eukaryote",
    "dbo:Fungus":                "dbo:Eukaryote",
    "dbo:Bacteria":              "dbo:Thing",
    "dbo:Archaea":               "dbo:Thing",

    # Animal subtypes
    "dbo:Insect":                "dbo:Animal",
    "dbo:Arachnid":              "dbo:Animal",
    "dbo:Crustacean":            "dbo:Animal",
    "dbo:Fish":                  "dbo:Animal",
    "dbo:Bird":                  "dbo:Animal",
    "dbo:Mammal":                "dbo:Animal",
    "dbo:Reptile":               "dbo:Animal",
    "dbo:Amphibian":             "dbo:Animal",
    "dbo:Mollusca":              "dbo:Animal",
    "dbo:RaceHorse":             "dbo:Mammal",

    # Plant subtypes
    "dbo:FloweringPlant":        "dbo:Plant",
    "dbo:Fern":                  "dbo:Plant",
    "dbo:Conifer":               "dbo:Plant",
    "dbo:ClubMoss":              "dbo:Plant",
    "dbo:Moss":                  "dbo:Plant",
    "dbo:GreenAlga":             "dbo:Plant",
    "dbo:Cycad":                 "dbo:Plant",
    "dbo:CultivatedVariety":     "dbo:Plant",
    "dbo:Grape":                 "dbo:FloweringPlant",

    # ── Work/Creative branch ──────────────────────────────────────────
    "dbo:Work":                  "dbo:Thing",
    "dbo:MusicalWork":           "dbo:Work",
    "dbo:Film":                  "dbo:Work",
    "dbo:TelevisionShow":        "dbo:Work",
    "dbo:TelevisionEpisode":     "dbo:TelevisionShow",
    "dbo:Software":              "dbo:Work",
    "dbo:ProgrammingLanguage":   "dbo:Work",
    "dbo:VideoGame":             "dbo:Software",
    "dbo:Book":                  "dbo:Work",
    "dbo:Novel":                 "dbo:Book",
    "dbo:Manga":                 "dbo:Book",
    "dbo:Magazine":              "dbo:Work",
    "dbo:Newspaper":             "dbo:Work",
    "dbo:RadioProgram":          "dbo:Work",
    "dbo:HollywoodCartoon":      "dbo:Work",
    "dbo:Game":                  "dbo:Work",
    "dbo:Album":                 "dbo:MusicalWork",
    "dbo:Single":                "dbo:MusicalWork",
    "dbo:AnimangaCharacter":     "dbo:FictionalCharacter",
    "dbo:FictionalCharacter":    "dbo:Work",
    "dbo:Website":               "dbo:Work",

    # ── Event branch ──────────────────────────────────────────────────
    "dbo:Event":                 "dbo:Thing",
    "dbo:SportsEvent":           "dbo:Event",
    "dbo:MilitaryConflict":      "dbo:Event",
    "dbo:OlympicEvent":          "dbo:SportsEvent",
    "dbo:OlympicResult":         "dbo:SportsEvent",
    "dbo:GrandPrix":             "dbo:SportsEvent",
    "dbo:WrestlingEvent":        "dbo:SportsEvent",
    "dbo:MixedMartialArtsEvent": "dbo:SportsEvent",
    "dbo:Election":              "dbo:Event",
    "dbo:FootballLeagueSeason":  "dbo:SportsEvent",
    "dbo:NationalFootballLeagueSeason": "dbo:FootballLeagueSeason",

    # ── Misc ──────────────────────────────────────────────────────────
    "dbo:MusicGenre":            "dbo:Genre",
    "dbo:Genre":                 "dbo:Thing",
    "dbo:Language":              "dbo:Thing",
    "dbo:EthnicGroup":           "dbo:Thing",
    "dbo:Food":                  "dbo:Thing",
    "dbo:Beverage":              "dbo:Food",
    "dbo:Award":                 "dbo:Thing",
    "dbo:Currency":              "dbo:Thing",
    "dbo:Colour":                "dbo:Thing",
    "dbo:Disease":               "dbo:Thing",
    "dbo:Automobile":            "dbo:Thing",
    "dbo:Aircraft":              "dbo:Thing",
    "dbo:Ship":                  "dbo:Thing",
    "dbo:Train":                 "dbo:Thing",
    "dbo:Weapon":                "dbo:Thing",
    "dbo:Drug":                  "dbo:Thing",
    "dbo:Holiday":               "dbo:Event",
    "dbo:Sport":                 "dbo:Thing",
    "dbo:ChemicalCompound":      "dbo:Thing",
    "dbo:Planet":                "dbo:NaturalPlace",
    "dbo:Nerve":                 "dbo:AnatomicalStructure",
    "dbo:AnatomicalStructure":   "dbo:Thing",
    "dbo:InformationAppliance":  "dbo:Device",
    "dbo:Device":                "dbo:Thing",
    "dbo:Agent":                 "dbo:Thing",
    "dbo:Eukaryote":             "dbo:Thing",
}


def _expand_ancestors(type_str: str, cache: dict) -> list[str]:
    """Return all ancestor types of `type_str` up to (but excluding) dbo:Thing."""
    if type_str in cache:
        return cache[type_str]
    parent = _DBO_PARENTS.get(type_str)
    if parent is None or parent == "dbo:Thing":
        ancestors = []
    else:
        ancestors = [parent] + _expand_ancestors(parent, cache)
    cache[type_str] = ancestors
    return ancestors


# Build the full ancestor list for every type that appears in data.
_cache: dict[str, list[str]] = {}
DBO_ANCESTORS: dict[str, list[str]] = {
    t: _expand_ancestors(t, _cache)
    for t in _DBO_PARENTS
}

# Add dbo:Thing itself as having no ancestors.
DBO_ANCESTORS["dbo:Thing"] = []


def get_all_types(leaf_type: str) -> list[str]:
    """Return [leaf_type] plus all its dbo: ancestor types."""
    if not leaf_type.startswith("dbo:"):
        return [leaf_type]  # non-dbo types (owl:Thing etc.) left as-is
    return [leaf_type] + DBO_ANCESTORS.get(leaf_type, [])


if __name__ == "__main__":
    # Quick sanity check
    for t in ["dbo:MusicalArtist", "dbo:Band", "dbo:SoccerClub",
              "dbo:Entomologist", "dbo:Settlement", "dbo:River"]:
        ancestors = get_all_types(t)
        print(f"  {t}: {' -> '.join(ancestors)}")
