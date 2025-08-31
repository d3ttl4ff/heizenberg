'use client';

import Image from 'next/image';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import React from 'react';

const logo = '/images/HELOGO6.png';
const navLinks = [{ label: 'About Heizenberg', path: '/about' }];

const Navbar = ({ scrollRef }: { scrollRef: React.RefObject<any> }) => {
  const pathname = usePathname();
  const router = useRouter();

  function handlePredictClick() {
    if (pathname === '/') {
      scrollRef?.current?.scrollTo?.('#predict', {
        offset: -50,
        duration: 1,
      });
    } else {
      router.push('/#predict');
    }
  }

  return (
    <div className="md:max-w-2xl sm:max-w-sm mx-auto w-fit fixed top-0 left-0 right-0 my-5 z-[50]">
      <div className="flex justify-center items-center bg-white/5 backdrop-blur-2xl text-center rounded-xl p-1">
        <div className="flex items-center justify-center gap-1 bg-white/5 rounded-lg p-1">
          <Link
            href="/"
            className="rounded-md px-4 py-2 relative h-9 w-9 overflow-hidden border border-white/5 bg-accent-nvidia-dim/5 hover:scale-95 active:scale-90 transition-all"
          >
            <Image src={logo} alt="Logo" fill className="object-contain" />
          </Link>

          <div className="rounded-md px-4 py-2 text-sm text-white/60 border border-white/5 hover:scale-[99%] transition-all">
            {navLinks.map((link, idx) => (
              <Link key={idx} href={link.path}>
                {link.label}
              </Link>
            ))}
          </div>

          <button
            type="button"
            className="rounded-md px-4 py-2 bg-accent-nvidia text-sm font-bold text-black/80 border border-white/5 cursor-pointer hover:bg-accent-nvidia-dim hover:scale-[99%] transition-all"
            onClick={handlePredictClick}
          >
            Predict Now
          </button>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
