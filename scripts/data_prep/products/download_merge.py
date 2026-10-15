#!/usr/bin/env python
"""
Download MERGE precipitation data from CPTEC/INPE.

MERGE: Multi-source Estimated Rain using Gauge observation
- Source: CPTEC/INPE
- Resolution: 0.1 degrees (~10km)
- Frequency: Hourly
- Coverage: South America

Usage:
    python scripts/download_merge.py --start 2021-01-01 --end 2025-12-31
    python scripts/download_merge.py --year 2021
    python scripts/download_merge.py --month 2021-06
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error
import time

# Configuracoes
BASE_URL = "https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/HOURLY"
OUTPUT_DIR = Path("data/raw/merge")
MAX_WORKERS = 4  # Downloads paralelos
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # segundos


def get_merge_url(dt: datetime) -> str:
    """Gera URL do arquivo MERGE para uma data/hora."""
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    filename = f"MERGE_CPTEC_{dt.strftime('%Y%m%d%H')}.grib2"
    return f"{BASE_URL}/{year}/{month}/{filename}", filename


def download_file(url: str, filepath: Path, retries: int = RETRY_ATTEMPTS) -> bool:
    """Baixa um arquivo com retry."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, filepath)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False  # Arquivo nao existe
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    return False


def download_hour(dt: datetime, output_dir: Path) -> tuple:
    """Baixa arquivo de uma hora especifica."""
    url, filename = get_merge_url(dt)
    filepath = output_dir / filename

    # Skip se ja existe
    if filepath.exists():
        return dt, "skip", 0

    success = download_file(url, filepath)
    if success:
        size = filepath.stat().st_size
        return dt, "ok", size
    else:
        return dt, "fail", 0


def generate_hours(start: datetime, end: datetime):
    """Gera todas as horas entre start e end."""
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=1)


def main():
    parser = argparse.ArgumentParser(description="Download MERGE precipitation data")
    parser.add_argument("--start", type=str, help="Data inicio (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Data fim (YYYY-MM-DD)")
    parser.add_argument("--year", type=int, help="Baixar ano inteiro")
    parser.add_argument("--month", type=str, help="Baixar mes (YYYY-MM)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Downloads paralelos")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria baixado")

    args = parser.parse_args()

    # Determinar periodo
    if args.year:
        start = datetime(args.year, 1, 1, 0)
        end = datetime(args.year, 12, 31, 23)
    elif args.month:
        year, month = map(int, args.month.split("-"))
        start = datetime(year, month, 1, 0)
        if month == 12:
            end = datetime(year, 12, 31, 23)
        else:
            end = datetime(year, month + 1, 1, 0) - timedelta(hours=1)
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23)
    else:
        print("Erro: Especifique --start/--end, --year ou --month")
        sys.exit(1)

    # Criar diretorio
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Listar horas
    hours = list(generate_hours(start, end))
    total_hours = len(hours)

    print("=" * 60)
    print("DOWNLOAD MERGE - CPTEC/INPE")
    print("=" * 60)
    print(f"Periodo: {start.strftime('%Y-%m-%d %H:%M')} a {end.strftime('%Y-%m-%d %H:%M')}")
    print(f"Arquivos: {total_hours:,}")
    print(f"Destino: {OUTPUT_DIR.absolute()}")
    print(f"Workers: {args.workers}")
    print()

    if args.dry_run:
        print("[DRY RUN] Nenhum arquivo sera baixado.")
        print(f"Primeiro: {get_merge_url(hours[0])[0]}")
        print(f"Ultimo:   {get_merge_url(hours[-1])[0]}")
        return

    # Download paralelo
    downloaded = 0
    skipped = 0
    failed = 0
    total_size = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_hour, dt, OUTPUT_DIR): dt for dt in hours}

        for i, future in enumerate(as_completed(futures), 1):
            dt, status, size = future.result()

            if status == "ok":
                downloaded += 1
                total_size += size
            elif status == "skip":
                skipped += 1
            else:
                failed += 1

            # Progresso
            if i % 100 == 0 or i == total_hours:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_hours - i) / rate if rate > 0 else 0

                print(f"\r[{i:,}/{total_hours:,}] "
                      f"OK:{downloaded:,} Skip:{skipped:,} Fail:{failed:,} "
                      f"| {total_size/1e9:.1f}GB "
                      f"| {rate:.1f}/s "
                      f"| ETA:{eta/60:.0f}min", end="", flush=True)

    print()
    print()
    print("=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Baixados:  {downloaded:,}")
    print(f"Existiam:  {skipped:,}")
    print(f"Falharam:  {failed:,}")
    print(f"Total:     {total_size/1e9:.2f} GB")
    print(f"Tempo:     {(time.time()-start_time)/60:.1f} min")

    if failed > 0:
        print(f"\n[AVISO] {failed} arquivos falharam. Execute novamente para retry.")


if __name__ == "__main__":
    main()
