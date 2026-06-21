import argparse
import collections
import logging
import random

from pachner_traversal.mcmc import iterate
from pachner_traversal.utils import data_root

logger = logging.getLogger(__name__)

seeds = [
    "pnlnpaaaiadjghnhmjmlllmoorqqcrlsoooowwwmj",
    "pnlnpaaaiadjilnmhnijlnloorxtthmdsivwwkwxf",
    "pnlnpaaaiadjjnikhmmnknloorxxhxahsqggcrsdv",
    "pnlnpaaaiadjknhlkliknlmoorxdqxraqijfbwcdw",
    "pnlnpaaaiadkhkmnmnjlljloorxmcmqedejwwwwpn",
    "pnlnpaaaiadkhlhkkimnllnoorxtclrlxvorvvoms",
    "pnlnpaaaiadknghnjjjlllmoorxxxcrxessrjsstw",
    "pnlnpaaaiadlgnnhmmiklklooruakxtwtvrnvnvmb",
    "pnlnpaaaiadlhmhkjnjjllnoorxxcmrtarsswwreb",
    "pnlfpjaaaacfigjmimlkonnnorsslcewsvsjmeeij",
    "pnlfpjaaaacgmhmkmnljonlnorcbxxlljxkwdpfpw",
    "pnlfpjaaaackgilkhmmnnlnoorbxraedvvvlmjder",
    "pnlfpjaaaadflmmjknnoonllorxjxxxeixwxqwwwi",
    "pnlfpjaaaadfmkglimjooonnnragxajxiorxllssr",
    "pnlfpjaaaadgllhiknjolmnnordfqqreinrtwrnnj",
    "pnlfpjaaaadhgmglknmlkmooorxksxbiqhbvjjtjs",
    "pnlfpjaaaadigjmlkinonnmoorxmsdqutprxssbxd",
    "pnlfpjaaaadjgmkimmnolnloorxtoxexxxwxvvstt",
    "pnlfpjaaaadjhkgmkljloomonrxxoqwtqwsvmispj",
    "pnlfpjaaaadjhmjljinlooonnrxucxttaacwmmhcc",
    "pnlfpjaaaadjlmlhjnnklomnorxxxxxrlxckopocc",
    "pnlfpjaaaadjmhjmhllmloonorxaxcaprffowmpcp",
    "pnlfpjaaaadkmhigmnklnoonorxxxcajuscvbttst",
    "pnlfpjaaaadlgjllhmjoknoonrxxsulewwsqvciij",
    "pnlfpjaaaadlgkhmhlknonmnorxxstcmscraipoxr",
    "pnlfpjaaaadlmimhinmkokoonrxxipiwuhwbxvuur",
    "pnlfpjaaaaefjkhkkmjnonoonrxfiqrqxwswlbltf",
    "pnlfpjaaaaeflmhkjiknnmmoorxnixrilvobfnnxs",
    "pnlfpjaaaaeglmllinmnkkmoorekmxmmixwrorgej",
    "pnlfpjaaaaehgmglljkomnonorekfxwmmscxvexxs",
    "pnlfpjaaaaehlmghljnonomonreklxcvlsxxxijxh",
    "pnlfpjaaaaeigihmhnnklnomoreqsirmsuuskuecv",
    "pnlfpjaaaaejgkilmnjnoolnorxxoqqxxxsvxxvsq",
    "pnlfpjaaaaejhhgjmnlklnoooreloowaaxvnnltts",
    "pnlfpjaaaaejhhgkmmmlkoonoreioowaaccrviicw",
    "pnlfpjaaaaejlmhghjkolmnoorxxixbjwscxofmcc",
    "pnlfpjaaaaejmghkhmmlolonorehqfrlsccsusucf",
    "pnlfpjaaaaekgjhjjinmoomonrxxstratjfcxiwiw",
    "pnlfpjaaaaekgjmihinlolnnorxqsetmsjkvtjoov",
    "pnlfpjaaaaelggmhmnjnknomorexsgqshxvuwlisv",
    "pnlfpjaaaaelggmmmillkooonredvveppjjjrmmej",
    "pnlfpjaaaaelghghmmnknnomorxxggjswwxrmmeos",
    "pnlfpjaaaaelhmghmjlokonnorxxwxcrxswxwtqhw",
    "pnlfpjaaaaellglhkmmonokonreqlglrqwwejesis",
    "pnlfpjaaaaellgmihmmlonoonrexmkmmwwrvxjxxj",
    "pnlfpfaaaadfmmhlmljonknnoragxxrtqrrlnvbor",
    "pnlfpfaaaadgkmhlkmnjomoonrxblxrtewwwtvttw",
    "pnlfpfaaaadhikjlhmolomonnrxcixmhrwuvxcxbr",
    "pnlfpfaaaadjgkmjiionmnonorxxsptdaaevvvevl",
    "pnlfpfaaaadjgmhmlnooknomnrxusxrdirxxbbakn",
    "pnlfpfaaaadjhhhgimllnomnorqqccwbjjwwharcr",
    "pnlfpfaaaadjhkgiikomnonmorxxwuwlurxnvxvsl",
    "pnlfpfaaaadjlmmhklomnnlnorxxlxerpdqbggwgk",
    "pnlfpfaaaadjmhhklmolknnoorqqxcsexwxvncfix",
    "pnlfpfaaaadkggmmminjoolonrxxssqudvvjmmoij",
    "pnlfpfaaaadkgmmljnjkolnoorxxsxtpaxsrtwsxi",
    "pnlfpfaaaadkljjhimkmolonorxxqimomxrcpsuch",
    "pnlfpfaaaadkmhghjnkollonorxmxccrhmbebbttc",
    "pnlfpfaaaadkmhlhmijolnnoorxxxclrevwqjrvix",
    "pnlfpfaaaadlgghkkljmomnoorxxssrixwwwukxjj",
    "pnlfpfaaaadlggmhjnmkloonorxxsshwetvjwmiij",
    "pnlfpfaaaadlhjghjmllomonorxxnacrpcrwmseco",
    "pnlfpfaaaadlhlgmkimomolonrxubdcadvvavpoxj",
    "pnlfpfaaaadlhlgmlimnmoonorxqbtcapvvovmmjm",
    "pnlfpfaaaadlijghhkmonoonnrxxxxcssvjdlddxx",
    "pnlfpfaaaadlkhmhlijokonnorxxpcmrpvwqrxgge",
    "pnlfpfaaaaefjihkmionomlnorevphraxjxbxgrsm",
    "pnlfpfaaaaeflmhlhnnoknomorxfpmrtsxxirmmgh",
    "pnlfpfaaaaegmhlljmooklnonrekxcmmmwxxcksij",
    "pnlfpfaaaaehggmmjnomllloorxvsshuxxhbsrrjj",
    "pnlfpfaaaaehgkhjjnkllonoorefgdreuxrkrdtdr",
    "pnlfpfaaaaehjhjhlmlklnnoorxvlclrmfcsbxxof",
    "pnlfpfaaaaehlhmhkjmonolonrxvdcurasjmamjdc",
    "pnlfpfaaaaehlmghlmmnkoonoreklxcvljjdrxxdk",
    "pnlfpfaaaaeigmmmhknoololnremsxtesokaakakc",
    "pnlfpfaaaaeihhhgknomnolmoretccnjwxxwuhkrc",
    "pnlfpfaaaaeihlgmhjonklnoorepomwpswxxvcumw",
    "pnlfpfaaaaeijghmhnlmnmoooreplcrxsxvjqjttw",
    "pnlfpfaaaaejghmmhnlkllonorxesrppwicrssdcs",
    "pnlfpfaaaaejgihjijlmnonoorxxkikpisgnbhcqi",
    "pnlfpfaaaaejgmkhmnokkolnorelsxqrxwxfrlrwi",
    "pnlfpfaaaaejihgmmnnkomoloreapcwxignrxftra",
    "pnlfpfaaaaekgjllmlokononnrxxsxetudxcxvevv",
    "pnlfpfaaaaekgmhminjlllnoorxxsqouprcnwwrhc",
    "pnlfpfaaaaekhhmhlmmnooolnrxxccurpsoppxpjs",
    "pnlfpfaaaaelgghhjkmllnnoorepskbvrcfkkuusj",
    "pnlfpfaaaaelgghijjommlonoretssrmesaffsxsn",
    "pnlfpfaaaaelghjllmkjmonoorexschaawcwjesii",
    "pnlfpfaaaaelgimihjnmmnooorexslttsscjjcxxt",
    "pnlfpfaaaaelgkhjjmnommlnorlusirdqwceccvjn",
    "pnlfpfaaaaelgmijllkmmnnoorexnxexhhckknwaw",
    "pnlfpfaaaaelgmkkmkjjloonorexsxiiuissvixcd",
    "pnlfpfaaaaelhhghimonlmoonrexocwnvkxukrxpw",
    "pnlfpfaaaaelhhgmmjnnkooonrexooweasppsexxt",
    "pnlfpfaaaaelhigikkomlmoonrexwqwxewxnknqir",
    "pnlfpfaaaaeljhkhkjjmlnonoretpchrhwwksctww",
    "pnlfpfaaaaellgmmhmjjklonorephkmmwwcjsbtjc",
    "pnlfpfaaaaellkghkiomknoonrexxlcrljmjrcttw",
    "pnlfpfaaaaelmhgkinnmklmoorxxxcnexvvbrjspn",
    "pnlfpfaaaaelmhhjhmjknloooretqcnqrcjsdsevv",
]


def generate_samples(leading_char, seed, gamma_, target_unique, steps):
    iso = seed
    discovered_isos = set()

    max_attempts = target_unique * 500
    attempts = 0

    while len(discovered_isos) < target_unique and attempts < max_attempts:
        iso = iterate(iso, gamma_, steps)

        if iso.startswith(leading_char):
            discovered_isos.add(iso)

        attempts += 1

    if attempts >= max_attempts:
        msg = f"Worker hit max attempts ({max_attempts}) before reaching target."
        logger.warning(msg)

    return list(discovered_isos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_id", type=int, default=0, help="Unique ID for this parallel worker"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of outer loops to run"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    gamma_ = 1 / 6
    target_unique = 100_000
    steps = 1
    size = 15
    leading_char = "p"

    seed_buffer = collections.deque(maxlen=100)
    seed_buffer.extend(seeds)

    save_path = data_root / "input_data" / "dehydration" / "raw" / "big_mcmc_samples"
    save_path.mkdir(parents=True, exist_ok=True)
    fname = f"samps_{size}_worker_{args.worker_id}.txt"

    with open(save_path / fname, "a") as f:
        for loop in range(args.iterations):
            logger.info(
                f"Worker {args.worker_id} starting batch {loop}/{args.iterations}"
            )
            seed = random.choice(tuple(seed_buffer))

            samples = generate_samples(
                leading_char=leading_char,
                seed=seed,
                gamma_=gamma_,
                target_unique=target_unique,
                steps=steps,
            )

            if samples:
                seed_buffer.extend(samples)
                f.writelines(f"{s}\n" for s in samples)
                f.flush()

    logger.info(f"Worker {args.worker_id} finished generation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
